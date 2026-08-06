"""Offline tests for the ``service_desk`` example — no LLM, no network.

Runs the supervisor ``route()`` flow against a mocked LLM transport to prove
the default :class:`draf.node.Supervisor` guards, and exercises the
knowledge base against a stub embedder (no network):

* the happy path — one supervisor call, then the ``done_keys`` guard
  finishes the turn deterministically;
* ``fallback_agent`` — a premature ``finish`` on an empty turn routes to the
  fallback specialist instead of ending silently;
* the deploy gateway — an :class:`draf.node.Interrupt` pauses inside the
  route chain and resumes with the operator's answer;
* multi-turn routing — a follow-up message resets the per-turn scratch and
  routes to a new specialist;
* knowledge base — search is scoped to one domain, tools are wired to their
  specialist.
"""

import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from service_desk.core.deps import build_deps  # noqa: E402
from service_desk.graphs.build import build_flow  # noqa: E402
from service_desk.graphs.state import STATE_REDUCERS, initial_state  # noqa: E402
from service_desk.storage import TRANSIENT_KEYS, build_checkpointer  # noqa: E402
from service_desk.tools.knowledge import SearchIncidentKnowledge  # noqa: E402

from draf import Assistant  # noqa: E402
from draf.node.interrupt import GraphInterrupt  # noqa: E402
from draf.rag.stores import InMemoryVectorStore  # noqa: E402


def _reply(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "message": {"role": "assistant", "content": content},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _stream_lines(content: str) -> list[str]:
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]})
        for chunk in (content[i : i + 4] for i in range(0, len(content), 4))
    ]
    lines.append("data: [DONE]")
    return lines


_BILLING = "Счёт оплачен, задолженность 0 ₽."
_INCIDENT = "Проверьте статус сервиса и перезапустите агента."
_DEPLOY = "1. Бэкап.\n2. Выкатка.\n3. Откат при сбое."
_FALLBACK = "Здравствуйте! Чем помочь?"
_FINAL = "Итоговый ответ сформирован."


class _MockTransport:
    """Serves a canned, system-prompt-aware reply per LLM call.

    The supervisor gets a scripted sequence of single words (one per model
    consultation); each specialist runs its canned content.  With the default
    ``done_keys`` guard the supervisor is consulted exactly once per turn.
    """

    def __init__(self, words: tuple[str, ...] = ("billing",)):
        self.supervisor_calls = 0
        self.words = list(words)

    def _content_for(self, body: dict) -> str:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        if "диспетчер службы поддержки" in system:
            self.supervisor_calls += 1
            return self.words.pop(0) if self.words else "finish"
        if "специалист по счетам" in system:
            return _BILLING
        if "специалист по инцидентам" in system:
            return _INCIDENT
        if "специалист по релизам" in system:
            return _DEPLOY
        if "дружелюбный ассистент" in system:
            return _FALLBACK
        if "финальный ответ" in system:
            return _FINAL
        return "Привет!"

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


class _StubEmbedder:
    """Deterministic fake embedder — 4 dims, no network."""

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            out.append([b / 255.0 for b in digest[:4]])
        return out


def _stub_knowledge():
    """A knowledge base seeded from the real CSVs, in-memory, offline."""
    return build_deps(embedder=_StubEmbedder(), store=InMemoryVectorStore(dim=4))


def _flow():
    """``build_flow`` bound to the offline knowledge base; returns (flow, tools)."""
    return build_flow(knowledge=_stub_knowledge())


def _state(message: str) -> dict:
    state = initial_state()
    state["messages"] = [{"role": "user", "content": message}]
    return state


def test_route_edges_exist():
    flow, _tools = _flow()
    graph = flow.compile()
    edges = {(e.source_id, e.target_id, e.condition) for e in graph.edges}

    assert ("reset", "supervisor", None) in edges
    assert ("supervisor", "billing", "next_agent=billing") in edges
    assert ("billing", "supervisor", None) in edges  # specialist loops back
    assert ("supervisor", "deploy", "next_agent=deploy") in edges
    assert ("deploy", "approve", None) in edges  # gateway inside the chain
    assert ("approve", "supervisor", None) in edges  # resume returns to decider
    assert ("supervisor", "final", "next_agent=finish") in edges
    assert graph.entry_point == "reset"


@pytest.mark.asyncio
async def test_billing_route_runs_and_finishes(transport):
    flow, tools = _flow()
    graph = flow.compile()

    result = await graph.run(
        _state("сколько на счёту?"),
        tools=tools,
        reducers=STATE_REDUCERS,
        max_iterations=80,
    )

    # one supervisor call; the done guard finishes the turn deterministically
    assert transport.supervisor_calls == 1
    assert result["billing"] == _BILLING
    assert result["supervisor_rounds"] == 2
    assert result["final"] == _FINAL

    # the specialist's reply was appended to the shared conversation
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    assert any(_BILLING in m["content"] for m in assistant)


@pytest.mark.asyncio
async def test_offtopic_routes_to_fallback(monkeypatch):
    mock = _MockTransport(words=("fallback",))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = _flow()
    result = await flow.compile().run(
        _state("привет"),
        tools=tools,
        reducers=STATE_REDUCERS,
        max_iterations=80,
    )

    assert result["fallback"] == _FALLBACK
    assert not result["billing"]
    assert mock.supervisor_calls == 1
    assert result["supervisor_rounds"] == 2


@pytest.mark.asyncio
async def test_premature_finish_routes_to_fallback(monkeypatch):
    """A stray 'finish' on an empty turn falls back instead of ending empty."""
    mock = _MockTransport(words=("finish",))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = _flow()
    result = await flow.compile().run(
        _state("привет"),
        tools=tools,
        reducers=STATE_REDUCERS,
        max_iterations=80,
    )

    assert result["fallback"] == _FALLBACK
    assert mock.supervisor_calls == 1
    assert result["supervisor_rounds"] == 2


@pytest.mark.asyncio
async def test_deploy_pauses_for_approval(monkeypatch, tmp_path):
    """The deploy gateway interrupts; the answer resumes to finish."""
    mock = _MockTransport(words=("deploy",))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = _flow()
    graph = flow.compile()
    cp = build_checkpointer(tmp_path)
    state = _state("выкати изменения в прод")
    sid = "deploy-run"

    with pytest.raises(GraphInterrupt) as exc:
        await graph.run(
            state,
            tools=tools,
            reducers=STATE_REDUCERS,
            checkpointer=cp,
            checkpoint_id=sid,
            max_iterations=80,
        )
    interrupt = exc.value
    assert interrupt.key == "deploy_approved"
    assert "Подтверждаешь выкатку" in interrupt.prompt

    done = await graph.run(
        state,
        tools=tools,
        reducers=STATE_REDUCERS,
        checkpointer=cp,
        checkpoint_id=sid,
        resume={"deploy_approved": "да"},
        max_iterations=80,
    )
    assert done["deploy"] == _DEPLOY
    assert done["deploy_approved"] == "да"
    assert done["supervisor_rounds"] == 2
    assert mock.supervisor_calls == 1  # resume finished via the done guard


@pytest.mark.asyncio
async def test_second_turn_routes_to_new_specialist(monkeypatch, tmp_path):
    """A follow-up message resets the per-turn scratch and routes afresh."""
    mock = _MockTransport(words=("billing", "incident"))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = _flow()
    assistant = Assistant(
        flow.compile(),
        tools,
        build_checkpointer(tmp_path),
        reducers=STATE_REDUCERS,
        initial_state=initial_state,
        transient_keys=TRANSIENT_KEYS,
    )

    first = await assistant.run("sess-1", "сколько на счёту?")
    assert first.state["billing"] == _BILLING

    second = await assistant.run("sess-1", "сайт недоступен, что делать?")
    assert second.state["incident"] == _INCIDENT
    # history is preserved and the supervisor routed again on the second turn
    users = [m for m in second.state["messages"] if m.get("role") == "user"]
    assert len(users) == 2
    assert mock.supervisor_calls == 2


# ---------------------------------------------------------------------------
# Knowledge base (offline, stub embedder)
# ---------------------------------------------------------------------------


async def _searched_domain(domain: str, query: str) -> str:
    return await _stub_knowledge().search(query, domain=domain)


@pytest.mark.asyncio
async def test_knowledge_search_returns_incident_rows():
    result = await _searched_domain("incidents", "ошибка 500 при оплате")
    assert result  # non-empty
    assert "bug-452" in result  # the incident resolution came through
    # isolation: billing-only facts must not leak into the incidents domain
    assert "Возврат" not in result


@pytest.mark.asyncio
async def test_knowledge_search_scoped_to_domain():
    billing = await _searched_domain("billing", "как оформить возврат")
    assert billing
    assert "refunds" in billing or "Возврат" in billing
    # an incidents query must not return billing rows
    incidents = await _searched_domain("incidents", "как оформить возврат")
    assert "refunds" not in incidents


@pytest.mark.asyncio
async def test_knowledge_tool_searches_its_domain():
    knowledge = _stub_knowledge()
    tool = SearchIncidentKnowledge(knowledge)
    assert tool.name == "search_incident_knowledge"
    assert await tool.arun("сайт недоступен", top_k=3)


@pytest.mark.asyncio
async def test_build_flow_wires_knowledge_tools(transport):
    flow, tools = _flow()
    names = {t.name for t in tools}
    assert names == {
        "search_incident_knowledge",
        "search_billing_knowledge",
        "search_deploy_knowledge",
    }

    # each specialist is scoped to a single knowledge tool via use_tools
    graph = flow.compile()
    for slot, expected in {
        "incident": "search_incident_knowledge",
        "billing": "search_billing_knowledge",
        "deploy": "search_deploy_knowledge",
    }.items():
        sub = graph.nodes[slot]
        allowed = {
            n.config.get("use_tools")
            for n in sub._graph.nodes.values()
            if n.config.get("use_tools")
        }
        assert allowed == {expected}


# ---------------------------------------------------------------------------
# FastAPI + observability (offline)
# ---------------------------------------------------------------------------


def _client(monkeypatch, tmp_path, words=("billing",)):
    """A TestClient backed by the mock transport, stub knowledge and a tmp
    checkpoint/trace store."""
    pytest.importorskip("fastapi")
    mock = _MockTransport(words=words)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)
    from fastapi.testclient import TestClient
    from service_desk.server import create_app

    client = TestClient(
        create_app(
            checkpoint_dir=str(tmp_path / "ckpt"),
            knowledge=_stub_knowledge(),
            traces_db=str(tmp_path / "traces.db"),
        ),
        raise_server_exceptions=False,
    )
    return client, mock


def test_api_health(monkeypatch, tmp_path):
    client, _mock = _client(monkeypatch, tmp_path)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "provider": "ollama", "model": "llama3.1:8b"}


def test_api_chat_routes_and_persists_trace(monkeypatch, tmp_path):
    client, mock = _client(monkeypatch, tmp_path)

    resp = client.post("/api/chat", json={"message": "сколько на счёту?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"]  # the billing specialist's answer flows through
    assert data["session_id"]
    run_id = data["run_id"]
    assert run_id  # observability captured the turn
    assert mock.supervisor_calls == 1  # the done guard finished after one call

    # the trace is browsable through the dashboard API
    runs = client.get("/obs/runs").json()
    assert runs["total"] >= 1
    assert any(bi["run_id"] == run_id for bi in runs["items"])

    detail = client.get(f"/obs/runs/{run_id}").json()
    assert detail["run_id"] == run_id
    assert detail["status"] == "ok"
    assert "billing" in {n["id"] for n in detail["topology"]["nodes"]}


def test_api_deploy_gateway_pauses_then_resumes(monkeypatch, tmp_path):
    client, _mock = _client(monkeypatch, tmp_path, words=("deploy",))

    first = client.post("/api/chat", json={"message": "выкати изменения в прод"})
    assert first.status_code == 200
    fdata = first.json()
    assert fdata["message"]  # the deploy-approval prompt is surfaced as waiting
    sid = fdata["session_id"]

    resumed = client.post("/api/chat", json={"message": "да", "session_id": sid})
    assert resumed.status_code == 200
    body = resumed.json()
    assert body["message"]  # final summary honours deploy_approved
    assert _DEPLOY in body["message"]


def test_api_streams_events(monkeypatch, tmp_path):
    client, _mock = _client(monkeypatch, tmp_path, words=("incident",))

    stream = client.post("/api/chat/stream", json={"message": "сайт недоступен"})
    assert stream.status_code == 200
    assert "event: chat_id" in stream.text
    assert "event: run_start" in stream.text
    assert "event: message" in stream.text
    assert "event: node_error" not in stream.text
