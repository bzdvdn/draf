"""Offline API tests — no LLM, no network.

Runs the FastAPI server against a mocked LLM transport to prove the
supervisor flow is served correctly: the ``/api/chat`` single-shot reply,
the SSE ``/api/chat/stream`` stream, the durable ``/api/runs`` session
endpoints, and the API-key gate.  The mock serves both the non-streaming
``post`` path and the SSE ``stream`` path used by ReAct agents with
streaming enabled.

Requires the optional ``api`` extra (``uv sync --extra api``); skipped
otherwise so ``uv run pytest tests/`` always works.
"""

import json

import httpx
import pytest

pytest.importorskip("fastapi")

from app import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from src.config.config import Settings  # noqa: E402


def _reply(content: str) -> dict:
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
    calls ``post(url, headers=..., json=...)`` (awaited) and
    ``stream("POST", url, headers=..., json=...)`` (used as an async
    context manager), so the mock dispatches on the first positional: a
    URL means ``post`` (return a coroutine), ``"POST"`` means ``stream``
    (return the async context manager synchronously).
    """

    def __init__(self, *, finish: bool = True):
        self.supervisor_calls = 0
        self.finish = finish

    def _content_for(self, body: dict) -> str:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        if "Supervisor" in system:
            self.supervisor_calls += 1
            if self.finish:
                return "planner" if self.supervisor_calls == 1 else "finish"
            return "planner"  # never finish — proves the loop budget
        if "Planner" in system:
            return "1. Research. 2. Draft. 3. Review."
        return "All correct."

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
def client(tmp_path, monkeypatch):
    transport = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", transport)
    monkeypatch.setattr(httpx.AsyncClient, "stream", transport)
    return TestClient(
        create_app(
            Settings(
                api_key="test-key",
                checkpoint_dir=str(tmp_path),
                traces_db=str(tmp_path / "traces.db"),
            ),
        ),
        raise_server_exceptions=False,
    )


_HEADERS = {"X-API-Key": "test-key", "X-User-Id": "u1"}


def test_chat_runs_agent_and_returns_state(client):
    resp = client.post("/api/chat", json={"message": "help me"}, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"]
    assert isinstance(data["reply"], str)
    assert data["reply"]
    assert isinstance(data["run_id"], str)
    assert data["run_id"]


def test_chat_stream_emits_sse_events(client):
    resp = client.post(
        "/api/chat/stream", json={"message": "help me"}, headers=_HEADERS
    )
    assert resp.status_code == 200
    body = resp.text
    assert "event: chat_id" in body
    assert "event: run_start" in body
    assert "event: node_start" in body
    assert "event: run_end" in body
    assert "event: message" in body
    assert '"reply"' in body


def test_trace_dashboard_served(client):
    """The trace dashboard is mounted (behind auth) and records chat runs."""
    ui = client.get("/obs/ui", headers=_HEADERS)
    assert ui.status_code == 200
    assert "teff traces" in ui.text

    client.post("/api/chat", json={"message": "help me"}, headers=_HEADERS)
    page = client.get("/obs/runs", headers=_HEADERS).json()
    assert page["total"] == 1
    assert page["items"][0]["name"] == "chat"


def test_trace_dashboard_requires_auth(client):
    assert client.get("/obs/ui").status_code == 401
    assert client.get("/obs/runs").status_code == 401


def test_runs_get_and_delete(client):
    created = client.post(
        "/api/chat", json={"message": "help me"}, headers=_HEADERS
    ).json()
    chat_id = created["session_id"]

    saved = client.get(f"/api/runs/{chat_id}", headers=_HEADERS)
    assert saved.status_code == 200
    assert "state" in saved.json()

    deleted = client.delete(f"/api/runs/{chat_id}", headers=_HEADERS)
    assert deleted.status_code == 200

    missing = client.get(f"/api/runs/{chat_id}", headers=_HEADERS)
    assert missing.status_code == 404


def test_health_reports_provider_and_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "ollama"
    assert body["model"]


def test_auth_verify_endpoint(client):
    resp = client.get("/api/auth/verify", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_auth_is_fail_closed(tmp_path, monkeypatch):
    """With no API key configured, protected routes 401 (no open default)."""
    transport = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", transport)
    monkeypatch.setattr(httpx.AsyncClient, "stream", transport)
    client = TestClient(
        create_app(checkpoint_dir=str(tmp_path)), raise_server_exceptions=False
    )
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/api/auth/verify").status_code == 401
    assert client.get("/obs/ui").status_code == 401


def test_sessions_require_user_id(client):
    """X-User-Id is mandatory — no silent DEFAULT_OWNER fallback."""
    resp = client.post(
        "/api/chat", json={"message": "hi"}, headers={"X-API-Key": "test-key"}
    )
    assert resp.status_code == 401


def test_supervisor_loop_is_bounded(tmp_path, monkeypatch):
    """A model that never says 'finish' must not hang the graph."""
    import asyncio

    from src.graphs.build import build_flow
    from src.graphs.state import STATE_REDUCERS, initial_state

    transport = _MockTransport(finish=False)
    monkeypatch.setattr(httpx.AsyncClient, "post", transport)
    monkeypatch.setattr(httpx.AsyncClient, "stream", transport)

    flow, tools = build_flow(model="gpt-4o")
    graph = flow.compile()
    state = initial_state()
    state["messages"] = [{"role": "user", "content": "help me"}]

    result = asyncio.run(
        graph.run(
            state,
            tools=tools,
            reducers=STATE_REDUCERS,
            max_iterations=100,
        )
    )
    # the loop budget forces finish without a 100-iteration hang
    assert result["supervisor_rounds"] <= 6
    assert transport.supervisor_calls < 100


def test_api_key_gates_routes_when_configured(tmp_path, monkeypatch):
    transport = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", transport)
    monkeypatch.setattr(httpx.AsyncClient, "stream", transport)
    app = create_app(Settings(api_key="s3cret"), checkpoint_dir=str(tmp_path))
    client = TestClient(app, raise_server_exceptions=False)

    denied = client.post("/api/chat", json={"message": "hi"})
    assert denied.status_code == 401

    ok = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"X-API-Key": "s3cret", "X-User-Id": "u1"},
    )
    assert ok.status_code == 200

    bad = client.get("/api/auth/verify", headers={"X-API-Key": "wrong"})
    assert bad.status_code == 401
