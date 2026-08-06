"""End-to-end wiring + API tests for the ``repair-ai-chat`` application.

Builds the supervisor flow from ``examples/applications/repair-ai-chat`` and runs
it against a mocked LLM transport — no network, no API keys — to prove the
``route()`` loop, tool scoping, ``SubFlow`` agent chains, streaming, the
final structured extraction, and the FastAPI server (chat, SSE, sessions)
all work together.
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "applications" / "repair-ai-chat"
)
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from src.graphs.build import build_flow  # noqa: E402
from src.graphs.state import STATE_REDUCERS, initial_state  # noqa: E402


def _stub_embedder(client) -> None:
    """Swap the app catalog's embedder for a deterministic offline stub."""

    async def _embed_many(texts):
        return [
            list(__import__("numpy").random.default_rng(sum(map(ord, t))).random(4))
            for t in texts
        ]

    catalog = client.app.state.catalog
    catalog.embedder = type("_Stub", (), {"embed_many": staticmethod(_embed_many)})()
    catalog.store = __import__(
        "draf.rag.stores", fromlist=["InMemoryVectorStore"]
    ).InMemoryVectorStore(dim=4)
    catalog._ingested = 0


def test_catalog_load_and_update(tmp_path):
    """POST /api/catalog/load ingests a CSV in batches; update rebuilds."""
    pytest.importorskip("fastapi")
    from app import create_app
    from fastapi.testclient import TestClient

    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "Наименование,Цена,Ед\nКирпич М-150,24.2,₽/шт\nПлитка Керама-Белый,780,₽/м²\n",
        encoding="utf-8",
    )

    client = TestClient(create_app(checkpoint_dir=str(tmp_path)))
    _stub_embedder(client)

    status = client.get("/api/catalog")
    assert status.status_code == 200
    queued0 = status.json()["queued"]
    assert status.json()["stored"] == 0  # stub cleared the store, nothing loaded

    loaded = client.post(
        "/api/catalog/load", data={"batch_size": "1", "path": str(csv_path)}
    )
    assert loaded.status_code == 200
    body = loaded.json()
    assert body["queued_this_file"] == 2
    assert body["report"]["stored"] == queued0 + 2
    assert body["report"]["batches"] >= 1  # batched embed_many calls
    assert body["report"]["added"] == queued0 + 2

    # loading the same file again appends its rows to the store
    again = client.post(
        "/api/catalog/load", data={"batch_size": "10", "path": str(csv_path)}
    )
    assert again.json()["queued_this_file"] == 2
    assert again.json()["report"]["stored"] == queued0 + 4

    # update rebuilds the whole store from every queued document
    updated = client.post("/api/catalog/update", data={"batch_size": "10"})
    assert updated.status_code == 200
    assert updated.json()["report"]["stored"] == queued0 + 4


def _reply(content: str) -> dict:
    """A response that satisfies both OpenAI (``choices``) and Ollama
    (root ``message``) extraction paths used by the framework."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "message": {"role": "assistant", "content": content},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


async def _run_with_approval(
    graph, state, tools, *, answers=("да",), emit=None, checkpoint_dir="/tmp"
):
    """Run the staged graph end to end, answering every approval interrupt.

    The graph pauses on the plan- and estimate-approval interrupts; *answers*
    supplies the operator's replies in order (e.g. ``("нет", "да")`` to force
    one re-plan / re-estimate round).  Uses a per-run JSON-file checkpointer
    so resume works.
    """
    from draf.checkpoint import JSONFileCheckpointer
    from draf.node.interrupt import GraphInterrupt

    answers = list(answers)
    cp = JSONFileCheckpointer(str(checkpoint_dir))
    try:
        return await graph.run(
            state,
            tools=tools,
            reducers=STATE_REDUCERS,
            checkpointer=cp,
            checkpoint_id="run-1",
            max_iterations=80,
            emit=emit,
        )
    except GraphInterrupt:
        pass
    while True:
        answer = answers.pop(0) if answers else "да"
        try:
            return await graph.run(
                state,
                tools=tools,
                reducers=STATE_REDUCERS,
                checkpointer=cp,
                checkpoint_id="run-1",
                max_iterations=80,
                emit=emit,
                resume={"plan_approved": answer, "estim_approved": answer},
            )
        except GraphInterrupt:
            continue


def _stream_lines(content: str) -> list[str]:
    """Split *content* into OpenAI-style SSE chunks."""
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]})
        for chunk in (content[i : i + 4] for i in range(0, len(content), 4))
    ]
    lines.append("data: [DONE]")
    return lines


class _MockTransport:
    """Serves a canned, system-prompt-aware reply per LLM call.

    Installed on both ``httpx.AsyncClient.post`` and ``.stream``.  httpx
    calls ``post(url, headers=..., json=...)`` and
    ``stream("POST", url, headers=..., json=...)``, so the mock dispatches
    on the first positional: a URL means ``post``, ``"POST"`` means
    ``stream``.
    """

    def __init__(self, *, finish: bool = True, qa_verdicts: list[str] | None = None):
        self.calls: list[str] = []
        self.supervisor_calls = 0
        self.qa_calls = 0
        self.finish = finish
        self.qa_verdicts = list(qa_verdicts or [])

    def _content_for(self, body: dict) -> str:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        self.calls.append(system[:40])
        if "Supervisor" in system:
            self.supervisor_calls += 1
            return "pipeline"  # one-shot entry router: always start the pipeline
        if "извлекаешь" in system:
            return json.dumps({"room_type": "bathroom", "area": 5.0})
        if "классифицируешь" in system:
            user = " ".join(
                m.get("content", "")
                for m in body.get("messages", [])
                if m.get("role") == "user"
            )
            approved = any(
                w in user.lower()
                for w in ("да", "конечно", "хорошо", "ок", "согласен", "давай")
            )
            return json.dumps({"ok": approved, "message": ""})
        if "Planner" in system:
            return "1. Демонтаж. 2. Стены. 3. Пол. 4. Отделка."
        if "Estimator" in system:
            return "Смета: стены 12000, пол 20000, отделка 30000."
        if "Materials Agent" in system:
            return "Плитка Керама-Белый 780 ₽/м², ламинат Дуб-Прованс 890 ₽/м²."
        if "QA Agent" in system:
            if self.qa_verdicts:
                verdict = self.qa_verdicts[
                    min(self.qa_calls, len(self.qa_verdicts) - 1)
                ]
                self.qa_calls += 1
                return verdict
            return json.dumps({"ok": True, "message": ""})
        return "Здравствуйте! Помогу спланировать ремонт."

    def __call__(self, *args, **kwargs):
        content = self._content_for(kwargs.get("json") or {})

        if args and args[0] == "POST":  # streaming path

            class _StreamResp:
                def raise_for_status(self):
                    pass

                async def aiter_lines(self):
                    for line in _stream_lines(content):
                        yield line

            class _StreamCM:
                async def __aenter__(self):
                    return _StreamResp()

                async def __aexit__(self, *exc):
                    return False

            return _StreamCM()

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return _reply(content)

        async def _post():
            return _Resp()

        return _post()


@pytest.fixture
def transport(monkeypatch):
    mock = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)
    return mock


@pytest.mark.asyncio
async def test_route_loop_runs_end_to_end(transport, tmp_path):
    flow, tools = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Спланируй ремонт ванной 5 м²."}]

    events = []

    async def sink(ev):
        events.append(ev)

    result = await _run_with_approval(
        graph, state, tools, emit=sink, checkpoint_dir=str(tmp_path)
    )

    # the one-shot supervisor consults the model exactly once, then the
    # staged pipeline planner -> (approval) -> estimator -> materials -> qa
    # runs; the approval interrupt is answered with "да"
    assert transport.supervisor_calls == 1
    for section in ("plan", "estimate", "material_findings"):
        assert result[section] != ""
    assert result["plan_approved"] == "да"
    assert result["qa_ok"] == "yes"
    assert result["qa_rounds"] == 1
    assert result["project_info"] == {"room_type": "bathroom", "area": 5.0}

    # the assistant reply was appended to the shared conversation
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    assert any("Отделка" in m["content"] for m in assistant)

    # the deterministic План/Смета answer is the last assistant message
    last = result["messages"][-1]["content"]
    assert result["messages"][-1]["role"] == "assistant"
    assert last.startswith("План:")
    assert "Смета:" in last

    # a single top-level lifecycle per run; each approval interrupt pauses
    # with an interrupt event and its resume completes the run
    types = [ev.type for ev in events]
    assert types.count("run_start") == 3  # initial + plan resume + estimate resume
    assert types.count("interrupt") == 2  # plan + estimate approval
    assert types.count("run_end") == 1
    assert events[-1].data["status"] == "ok"

    # the decider and the routed agents all reported as nodes
    node_types = {ev.node_type for ev in events if ev.type == "node_start"}
    assert "supervisor" in node_types
    assert "subflow" in node_types


@pytest.mark.asyncio
async def test_staged_pipeline_replans_on_rejection(transport, tmp_path):
    """A "нет" answer re-runs the planner and re-asks; "да" then completes."""
    flow, tools = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Спланируй ремонт ванной 5 м²."}]

    result = await _run_with_approval(
        graph,
        state,
        tools,
        answers=("нет", "да"),
        checkpoint_dir=str(tmp_path),
    )

    assert result["plan_approved"] == "да"
    assert result["plan"] != ""
    assert result["estimate"] != ""
    assert result["material_findings"] != ""
    assert result["qa_ok"] == "yes"

    # the planner ran twice (initial + rework after "нет")
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    planner_replies = [
        m["content"] for m in assistant if m["content"].startswith("1. Демонтаж.")
    ]
    assert len(planner_replies) == 2

    last = result["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"].startswith("План:")
    assert "Смета:" in last["content"]


@pytest.mark.asyncio
async def test_estimate_rejection_recalculates_and_reasks(transport, tmp_path):
    """A "нет" on the estimate re-runs estimator/materials/QA before asking
    again; the second "да" then completes and assembles the final answer."""
    flow, tools = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Спланируй ремонт ванной 5 м²."}]

    # plan: да; estimate: нет then да (one recalculation round)
    result = await _run_with_approval(
        graph,
        state,
        tools,
        answers=("да", "нет", "да"),
        checkpoint_dir=str(tmp_path),
    )

    assert result["plan_approved"] == "да"
    assert result["estim_approved"] == "да"
    assert result["est_ok"] == "да"
    assert result["plan"] != ""
    assert result["estimate"] != ""
    assert result["material_findings"] != ""
    assert result["qa_ok"] == "yes"

    # the planner ran once (plan approved on the first ask); the estimator ran
    # twice (initial + recalc after "нет" on the estimate)
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    planner_replies = [
        m["content"] for m in assistant if m["content"].startswith("1. Демонтаж.")
    ]
    assert len(planner_replies) == 1
    # QA ran after the initial estimate and again after the recalc
    assert sum(1 for c in transport.calls if "QA Agent" in c) >= 2

    last = result["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"].startswith("План:")
    assert "Смета:" in last["content"]


@pytest.mark.asyncio
async def test_qa_fix_loop_revises_and_finalizes(monkeypatch, tmp_path):
    """A not-ok QA verdict re-runs estimator/materials, then the
    loop terminates on the ok verdict and appends the assembled answer."""
    mock = _MockTransport(
        finish=True,
        qa_verdicts=[
            json.dumps({"ok": False, "message": "Смета не сходится с планом."}),
            json.dumps({"ok": True, "message": ""}),
        ],
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Спланируй ремонт ванной 5 м²."}]

    result = await _run_with_approval(graph, state, tools, checkpoint_dir=str(tmp_path))

    assert mock.qa_calls == 2
    assert result["qa_rounds"] == 2
    assert result["qa_ok"] == "yes"

    # the fix body re-ran estimator/materials (planner is *not* re-run —
    # the plan was already approved), so the planner answered exactly once
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    planner_replies = [
        m["content"] for m in assistant if m["content"].startswith("1. Демонтаж.")
    ]
    assert len(planner_replies) == 1

    # the deterministic answer is the last assistant message
    last = result["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"].startswith("План:")
    assert "Смета:" in last["content"]


def test_project_info_schema_allows_null_for_unknown_fields():
    """Extractor prompt tells the model to return ``null`` for missing
    fields; the schema must accept that (regression: number fields rejected
    ``null``, so the extractor burned 3 attempts then raised NodeError)."""
    from src.graphs.schemas import PROJECT_INFO_SCHEMA

    from draf.schema import validate_json

    unknown = {
        "room_type": "bathroom",
        "area": 5.0,
        "ceiling_height": None,
        "budget": None,
        "style": None,
        "walls_area": None,
        "floor_area": None,
        "ceiling_area": None,
    }
    assert validate_json(unknown, PROJECT_INFO_SCHEMA) == []
    full = {
        "room_type": "bathroom",
        "area": 5.0,
        "ceiling_height": 2.7,
        "budget": 150000,
        "style": "modern",
        "walls_area": 20.0,
        "floor_area": 5.0,
        "ceiling_area": 5.0,
    }
    assert validate_json(full, PROJECT_INFO_SCHEMA) == []


def test_detect_room_type_maps_russian_keywords():
    from src.nodes.extractor import detect_room_type

    assert detect_room_type("Помоги спланировать ремонт ванной комнаты") == "bathroom"
    assert detect_room_type("Сделайте ремонт на кухне") == "kitchen"
    assert detect_room_type("нужен ремонт спальни") == "bedroom"
    assert detect_room_type("гостиная 20 метров") == "living_room"
    assert detect_room_type("прихожая маленькая") == "hallway"
    assert detect_room_type("какой стиль выбрать?") is None


def test_find_similar_accepts_materials_and_category_kwargs():
    """llama3.1:8b calls find_similar_material with ``materials`` and
    ``category`` kwargs; the tool must accept them instead of raising
    ``unexpected keyword argument`` (regression from obs run 99)."""
    from src.tools.rag import FindSimilarMaterial

    calls = []

    class _Catalog:
        async def find_similar(self, name, top_k=3):
            calls.append((name, top_k))
            return f"similar:{name}"

    tool = FindSimilarMaterial(_Catalog())
    result = asyncio.run(tool.arun(materials=["грунтовка"], category="кухня", top_k=3))
    assert result == "similar:грунтовка"
    assert calls == [("грунтовка", 3)]

    result = asyncio.run(tool.arun(name="краска", top_k=5))
    assert result == "similar:краска"
    assert calls[-1] == ("краска", 5)


@pytest.mark.asyncio
async def test_search_materials_accepts_string_max_price():
    """A string max_price (e.g. ``"2000"``) must be coerced to float and not
    blow up with ``'<=' not supported between instances of 'float' and 'str'``."""
    import json as _json

    from src.tools.rag import SearchMaterials

    from draf.harness.tools import execute_tool_calls

    seen = {}

    class _Catalog:
        async def search(self, query, category=None, max_price=None):
            seen.update(query=query, category=category, max_price=max_price)
            return "found"

    tool = SearchMaterials(_Catalog())
    result = await execute_tool_calls(
        [
            {
                "id": "call_1",
                "name": "search_materials",
                "args": _json.dumps({"query": "дверь", "max_price": "2000"}),
            }
        ],
        {"search_materials": tool},
    )
    assert result == ["found"]
    assert isinstance(seen["max_price"], float)
    assert seen["max_price"] == 2000.0


@pytest.mark.asyncio
async def test_search_retries_without_category_filter(monkeypatch):
    """A room-type category (``кухня``) matches nothing; the catalog must
    retry without the filter so the agent gets results instead of a dead
    ``Nothing found`` loop."""
    from src.rag.catalog import MaterialCatalog

    calls = []

    class _Store:
        def __init__(self):
            self.calls = calls

        async def search(self, query_vector, k=3, filter=None):
            calls.append(filter)
            if filter:
                return []
            return [
                (0, 0.9, {"name": "Грунтовка", "price": 200, "category": "грунтовка"})
            ]

    class _Embedder:
        async def embed(self, text):
            return [0.0] * 4

    catalog = MaterialCatalog(_Embedder(), store=_Store())
    catalog._docs = [("грунтовка", {})]
    catalog._ingested = 1

    result = await catalog.search("грунтовка", category="кухня", max_price=500)
    assert "Грунтовка" in result
    assert calls == [{"category": "кухня"}, None]


@pytest.mark.asyncio
async def test_extractor_falls_back_when_model_drops_room_type(monkeypatch):
    """llama3.1:8b often omits room_type; the node must fill it from the
    first user message so downstream agents see the room."""
    from src.nodes.extractor import Extractor

    from draf.node import LLM

    async def fake_llm_execute(self, ctx, state):
        return {"project_info": {"area": 5.0}}

    monkeypatch.setattr(LLM, "execute", fake_llm_execute)
    result = await Extractor().execute(
        None,
        {
            "messages": [
                {"role": "user", "content": "Помоги спланировать ремонт ванной 5 м²."},
                {"role": "assistant", "content": "План готов."},
            ]
        },
    )
    assert result["project_info"]["room_type"] == "bathroom"
    assert result["project_info"]["area"] == 5.0


@pytest.mark.asyncio
async def test_route_loop_survives_null_project_info(transport, tmp_path):
    """A real llama3.1:8b often returns ``null`` for fields it cannot find
    (ceiling_height, walls_area, ceiling_area); the run must not 500."""
    flow, tools = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Спланируй ремонт ванной 5 м²."}]

    result = await _run_with_approval(graph, state, tools, checkpoint_dir=str(tmp_path))
    assert result["supervisor_rounds"] <= 6
    assert result["project_info"]["area"] == 5.0


@pytest.mark.asyncio
async def test_route_wiring_in_example(transport):
    """The example's flow carries the staged, interrupt-driven edges."""
    flow, _ = build_flow()
    graph = flow.compile()
    edges = {(e.source_id, e.target_id, e.condition) for e in graph.edges}

    assert ("supervisor", "direct", "next_agent=direct") in edges
    assert ("supervisor", "pipeline-planner", "next_agent=pipeline") in edges
    # the plan-approval trio sits between the planner and the plan decider
    assert ("pipeline-planner", "plan-approval-interrupt", None) in edges
    assert ("plan-approval-interrupt", "plan-approval-classifier", None) in edges
    assert ("plan-approval-classifier", "plan-approval-validate", None) in edges
    # the plan validate decider routes on the *normalized* decision key
    assert any(
        src == "plan-approval-validate" and cond == "plan_ok=да" for src, _, cond in edges
    )
    assert any(
        src == "plan-approval-validate" and cond == "plan_ok!=да" for src, _, cond in edges
    )
    # approving the plan collects project info via the extractor
    assert any(
        src == "plan-approval-validate" and cond == "plan_ok=да"
        for src, target, cond in edges
    )
    # the "нет" branch re-plans (loop body copy) and loops back to the decider
    assert any(
        target == "plan-approval-validate" and cond is None for src, target, cond in edges
    )
    # once the plan is approved: extractor -> estimator -> materials -> qa -> estimate loop
    assert any(
        src == "estimator" and target == "materials" and cond is None
        for src, target, cond in edges
    )
    assert ("qa-subflow", "est-approval-interrupt", None) in edges
    # the estimate-approval trio sits between QA and the estimate decider
    assert ("est-approval-interrupt", "est-approval-classifier", None) in edges
    assert ("est-approval-classifier", "est-approval-validate", None) in edges
    # the estimate validate decider routes on its own normalized decision key
    assert any(
        src == "est-approval-validate" and cond == "est_ok=да" for src, _, cond in edges
    )
    assert any(
        src == "est-approval-validate" and cond == "est_ok!=да" for src, _, cond in edges
    )
    # a "нет" on the estimate re-runs estimator/materials/QA and loops back
    assert any(
        target == "est-approval-validate" and cond is None for src, target, cond in edges
    )
    # node types are present across the whole staged graph
    node_types = {n.type for nid, n in graph.nodes.items()}
    assert "extractor" in node_types


@pytest.mark.asyncio
async def test_supervisor_loop_is_bounded(monkeypatch, tmp_path):
    """The plan-approval fixture terminates when the user keeps answering
    "нет" — the interrupt re-asks without hanging the example graph."""
    from src.graphs.state import initial_state

    mock = _MockTransport(finish=False)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Спланируй ремонт ванной 5 м²."}]

    # the supervisor is a one-shot router — never a multi-round loop
    result = await _run_with_approval(graph, state, tools, checkpoint_dir=str(tmp_path))
    assert result["supervisor_rounds"] <= 6
    assert mock.supervisor_calls == 1


@pytest.mark.asyncio
async def test_catalog_reingest_task_detects_changes(tmp_path, monkeypatch):
    """The beat re-ingest rebuilds only when a seed CSV's content changes."""
    from src.queue.ingest import _fingerprint, reingest_if_changed

    catalog = __import__(
        "src.rag.catalog", fromlist=["MaterialCatalog"]
    ).MaterialCatalog
    store = __import__(
        "draf.rag.stores", fromlist=["InMemoryVectorStore"]
    ).InMemoryVectorStore(dim=4)

    async def _embed_many(texts):
        return [
            list(__import__("numpy").random.default_rng(sum(map(ord, t))).random(4))
            for t in texts
        ]

    stub = type("_Stub", (), {"embed_many": staticmethod(_embed_many)})()
    (tmp_path / "prices.csv").write_text(
        "Наименование,Цена,Ед\nКирпич,10,₽/шт\n", encoding="utf-8"
    )
    cat = catalog(embedder=stub, store=store)
    cat.add_csv(
        str(tmp_path / "prices.csv"),
        fieldmap={"name": "Наименование", "price": "Цена", "unit": "Ед"},
    )

    state_file = tmp_path / "ingest_state.json"
    monkeypatch.setattr("src.queue.ingest._state_path", lambda: state_file)
    monkeypatch.setattr("src.queue.ingest.DEFAULT_CATALOG", tmp_path / "prices.csv")
    monkeypatch.setattr("src.queue.ingest.DEFAULT_PRICE_LIST", tmp_path / "prices.csv")

    # first run embeds; fingerprint recorded
    first = await reingest_if_changed(catalog=cat)
    assert first["status"] == "ok"
    assert first["stored"] == 1  # the single row was embedded
    assert state_file.exists()

    # unchanged source -> no-op, no duplicate work
    again = await reingest_if_changed(catalog=cat)
    assert again["status"] == "unchanged"

    # changed source -> rebuild happens again
    (tmp_path / "prices.csv").write_text(
        "Наименование,Цена,Ед\nКирпич,15,₽/шт\n", encoding="utf-8"
    )
    changed = await reingest_if_changed(catalog=cat)
    assert changed["status"] == "ok"
    assert _fingerprint([tmp_path / "prices.csv"]) != ""


def test_queue_fingerprint_tracks_content():
    """Fingerprint is content-based, not path-based."""
    from src.queue.ingest import _fingerprint

    a = __import__("tempfile").NamedTemporaryFile("w", delete=False, suffix=".csv")
    a.write("a,b\n1,2\n")
    a.close()
    b = __import__("tempfile").NamedTemporaryFile("w", delete=False, suffix=".csv")
    b.write("a,b\n1,3\n")
    b.close()

    assert _fingerprint([a.name, b.name]) != _fingerprint([b.name, a.name])
    assert _fingerprint([a.name]) != _fingerprint([b.name])
    assert _fingerprint([a.name]) == _fingerprint([a.name])


def test_api_chat_and_stream(transport, tmp_path):
    """The FastAPI server serves chat + SSE and persists sessions."""
    pytest.importorskip("fastapi")
    from app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(
        create_app(checkpoint_dir=str(tmp_path)), raise_server_exceptions=False
    )

    created = client.post(
        "/api/chat", json={"message": "Помоги спланировать ремонт ванной 5 м²."}
    )
    assert created.status_code == 200
    data = created.json()
    assert isinstance(data["message"], str)
    assert data["message"]
    assert isinstance(data["run_id"], str)
    assert data["run_id"]
    # the graph pauses on the plan-approval interrupt waiting for the user
    assert data["waiting"] is True
    assert data["prompt"]  # the approval question is surfaced to the client
    chat_id = data["session_id"]

    # resuming the same session with "да" approves the plan, then the estimate
    # approval pauses again; answering "да" once more completes the pipeline
    resumed = client.post(
        "/api/chat",
        json={"message": "да", "session_id": chat_id},
    )
    assert resumed.status_code == 200
    rdata = resumed.json()
    assert rdata["waiting"] is True  # waiting on the estimate-approval interrupt

    resumed = client.post(
        "/api/chat",
        json={"message": "да", "session_id": chat_id},
    )
    assert resumed.status_code == 200
    rdata = resumed.json()
    assert rdata["waiting"] is False
    assert rdata["message"].startswith("План:")
    assert "Смета:" in rdata["message"]

    stream = client.post(
        "/api/chat/stream", json={"message": "Помоги спланировать ремонт ванной 5 м²."}
    )
    assert stream.status_code == 200
    assert "event: chat_id" in stream.text
    assert "event: run_start" in stream.text
    assert "event: waiting" in stream.text
    assert "event: run_end" not in stream.text

    saved = client.get(f"/api/runs/{chat_id}")
    assert saved.status_code == 200
    assert "state" in saved.json()

    deleted = client.delete(f"/api/runs/{chat_id}")
    assert deleted.status_code == 200
