"""End-to-end wiring + API tests for the ``repair-ai-chat`` application.

Builds the supervisor flow from ``examples/applications/repair-ai-chat`` and runs
it against a mocked LLM transport — no network, no API keys — to prove the
``route()`` loop, tool scoping, ``SubFlow`` agent chains, streaming, the
final structured extraction, and the FastAPI server (chat, SSE, sessions)
all work together.
"""

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

    def __init__(self, *, finish: bool = True):
        self.calls: list[str] = []
        self.supervisor_calls = 0
        self.finish = finish

    def _content_for(self, body: dict) -> str:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        self.calls.append(system[:40])
        if "Supervisor" in system:
            self.supervisor_calls += 1
            if self.finish:
                return "planner" if self.supervisor_calls == 1 else "finish"
            return "planner"  # never finish — proves the loop budget
        if "анализируешь" in system:
            return json.dumps({"room_type": "bathroom", "area": 5.0})
        if "Planner" in system:
            return "1. Демонтаж. 2. Стены. 3. Пол. 4. Отделка."
        if "Estimator" in system:
            return "Смета: стены 12000, пол 20000, отделка 30000."
        if "Materials Agent" in system:
            return "Плитка Керама-Белый 780 ₽/м², ламинат Дуб-Прованс 890 ₽/м²."
        if "QA Agent" in system:
            return "Всё корректно."
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
async def test_route_loop_runs_end_to_end(transport):
    flow, tools = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Спланируй ремонт ванной 5 м²."}]

    events = []

    async def sink(ev):
        events.append(ev)

    result = await graph.run(
        state,
        tools=tools,
        reducers=STATE_REDUCERS,
        emit=sink,
        max_iterations=80,
    )

    # planner runs once, then the done guard (done_mode="any") finishes the
    # turn deterministically without another supervisor LLM call
    assert transport.supervisor_calls == 1
    assert result["plan"] != ""
    assert result["project_info"] == {"room_type": "bathroom", "area": 5.0}

    # the assistant reply was appended to the shared conversation
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    assert any("Отделка" in m["content"] for m in assistant)

    # a single top-level lifecycle; nested run_start/run_end are stripped
    types = [ev.type for ev in events]
    assert types.count("run_start") == 1
    assert types.count("run_end") == 1
    assert events[-1].data["status"] == "ok"

    # the decider and the routed agents all reported as nodes
    node_types = {ev.node_type for ev in events if ev.type == "node_start"}
    assert "supervisor" in node_types
    assert "subflow" in node_types


@pytest.mark.asyncio
async def test_route_wiring_in_example(transport):
    """The example's flow carries the expected supervisor edges."""
    flow, _ = build_flow()
    edges = {(e.source_id, e.target_id, e.condition) for e in flow.compile().edges}

    assert ("supervisor_1", "extractor_2", "next_agent=finish") in edges
    assert ("supervisor_1", "subflow_4", "next_agent=planner") in edges
    assert ("supervisor_1", "subflow_5", "next_agent=estimator") in edges
    assert ("supervisor_1", "subflow_7", "next_agent=qa") in edges
    assert ("subflow_4", "supervisor_1", None) in edges


@pytest.mark.asyncio
async def test_supervisor_loop_is_bounded(monkeypatch):
    """A model that never says 'finish' must not hang the example graph."""
    from src.graphs.state import STATE_REDUCERS, initial_state

    mock = _MockTransport(finish=False)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Спланируй ремонт ванной 5 м²."}]

    result = await graph.run(
        state,
        tools=tools,
        reducers=STATE_REDUCERS,
        max_iterations=80,
    )
    # the loop budget forces finish without an 80-iteration hang
    assert result["supervisor_rounds"] <= 6
    assert mock.supervisor_calls < 80


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
    assert data["result"]["project_info"] == {"room_type": "bathroom", "area": 5.0}
    chat_id = data["session_id"]

    stream = client.post(
        "/api/chat/stream", json={"message": "Помоги спланировать ремонт ванной 5 м²."}
    )
    assert stream.status_code == 200
    assert "event: chat_id" in stream.text
    assert "event: run_start" in stream.text
    assert "event: run_end" in stream.text

    saved = client.get(f"/api/runs/{chat_id}")
    assert saved.status_code == 200
    assert "state" in saved.json()

    deleted = client.delete(f"/api/runs/{chat_id}")
    assert deleted.status_code == 200
