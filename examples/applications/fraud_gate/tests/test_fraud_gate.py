"""Offline tests for the ``fraud_gate`` example — no real LLM, no network.

Runs the full graph (analyser LLM + router + optional review + finalizer LLM)
against a mocked HTTP transport that serves a canned, system-prompt-aware
reply, to prove the :class:`draf.node.Command` routing:

* **approve** — the router returns ``Command(goto="finalize")`` and the run
  finishes with the finalizer LLM producing the summary;
* **mid-risk review** — the router returns ``Command(update=...)`` (no
  ``goto``), the run pauses on the :class:`draf.node.Interrupt` and resumes
  with the analyst's answer; the finalizer runs afterwards;
* **deny** — the router returns ``Command(goto=Command.STOP)``; the run ends
  immediately, ``finalize`` is **never** reached.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fraud_gate.graphs.build import build_flow  # noqa: E402
from fraud_gate.graphs.state import STATE_REDUCERS  # noqa: E402
from fraud_gate.storage import build_checkpointer  # noqa: E402

from draf.node.interrupt import GraphInterrupt  # noqa: E402


def _reply(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "message": {"role": "assistant", "content": content},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class _MockTransport:
    """Serves a canned, system-prompt-aware reply per LLM node."""

    def __init__(
        self,
        risk: float,
        signals: str = "тестовые сигналы",
        narrative: str = "Итог сформирован.",
    ):
        self.calls: list[str] = []
        self.risk = risk
        self.signals = signals
        self.narrative = narrative

    def _content_for(self, body: dict) -> str:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        if "аналитик платёжного шлюза" in system:
            self.calls.append("analyser")
            return json.dumps({"risk": self.risk, "signals": self.signals})
        if "голос платёжного шлюза" in system:
            self.calls.append("finalizer")
            return json.dumps({"decision": "approve", "summary": self.narrative})
        self.calls.append("unknown")
        return ""

    def __call__(self, *args, **kwargs):
        content = self._content_for(kwargs.get("json") or {})

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
    def _install(mock):
        monkeypatch.setattr(httpx.AsyncClient, "post", mock)
        return mock

    return _install


def _tx(**kwargs) -> str:
    tx = {"id": "tx-1", "amount": 100, "merchant": "fake", "note": "покупка"}
    tx.update(kwargs)
    return json.dumps(tx)


def _state(message: str) -> dict:
    return {"messages": [{"role": "user", "content": message}]}


@pytest.mark.asyncio
async def test_approve_routes_to_finalizer(transport):
    mock = transport(_MockTransport(0.2))
    result = (
        await build_flow()
        .compile()
        .run(
            _state(_tx()),
            reducers=STATE_REDUCERS,
            max_iterations=20,
        )
    )
    assert mock.calls == ["analyser", "finalizer"]
    assert result["decision"] == "approve"
    assert result["final"] == {"decision": "approve", "summary": "Итог сформирован."}


@pytest.mark.asyncio
async def test_mid_risk_review_pauses_then_finalizes(transport, tmp_path):
    mock = transport(_MockTransport(0.7))
    graph = build_flow().compile()
    cp = build_checkpointer(tmp_path)

    with pytest.raises(GraphInterrupt) as exc:
        await graph.run(
            _state(_tx()),
            reducers=STATE_REDUCERS,
            checkpointer=cp,
            checkpoint_id="review-1",
            max_iterations=20,
        )
    assert exc.value.key == "review_decision"
    assert "проверки" in exc.value.prompt
    assert mock.calls == ["analyser"]  # finalizer not reached yet

    result = await graph.run(
        _state(_tx()),
        reducers=STATE_REDUCERS,
        checkpointer=cp,
        checkpoint_id="review-1",
        resume={"review_decision": "pass"},
        max_iterations=20,
    )
    assert result["decision"] == "review"
    assert mock.calls == ["analyser", "finalizer"]


@pytest.mark.asyncio
async def test_deny_stops_before_finalizer(transport):
    mock = transport(_MockTransport(0.95))
    result = (
        await build_flow()
        .compile()
        .run(
            _state(_tx()),
            reducers=STATE_REDUCERS,
            max_iterations=20,
        )
    )
    assert mock.calls == ["analyser"]  # finalizer never runs on deny
    assert result["decision"] == "deny"
    assert result["final"]["decision"] == "deny"
    assert "Отказ" in result["final"]["summary"]


# ---------------------------------------------------------------------------
# FastAPI (offline)
# ---------------------------------------------------------------------------


def _client(monkeypatch, tmp_path, risk=0.2):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fraud_gate.config.config import Settings
    from fraud_gate.server import create_app

    monkeypatch.setattr(httpx.AsyncClient, "post", _MockTransport(risk))
    client = TestClient(
        create_app(
            Settings(checkpoint_dir=str(tmp_path / "ckpt"), model="llama3.1:8b"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            traces_db=str(tmp_path / "traces.db"),
        ),
        raise_server_exceptions=False,
    )
    return client


def test_api_health(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    assert client.get("/api/health").json() == {
        "status": "ok",
        "provider": "ollama",
        "model": "llama3.1:8b",
    }


def test_api_approves_transaction(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, risk=0.2)
    resp = client.post(
        "/api/review",
        json={"tx": {"id": "tx-1", "amount": 100, "note": "покупка"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "approve"
    assert body["waiting"] is False
    assert body["final"]["decision"] == "approve"


def test_api_denies_transaction(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, risk=0.95)
    body = client.post("/api/review", json={"tx": {"id": "tx-2", "amount": 100}}).json()
    assert body["decision"] == "deny"
    assert body["waiting"] is False
    assert body["final"]["decision"] == "deny"
    assert "Отказ" in body["final"]["summary"]


def test_api_review_pauses_then_resumes(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, risk=0.7)
    first = client.post("/api/review", json={"tx": {"id": "tx-3", "amount": 100}})
    assert first.status_code == 200
    fdata = first.json()
    assert fdata["decision"] == "review"
    assert fdata["waiting"] is True
    assert fdata["prompt"]
    sid = fdata["session_id"]

    resumed = client.post(f"/api/review/{sid}/decide", json={"answer": "pass"})
    assert resumed.status_code == 200
    body = resumed.json()
    assert body["decision"] == "review"
    assert body["waiting"] is False
    assert body["final"]


def test_api_decide_rejects_non_pending_session(monkeypatch, tmp_path):
    """Deciding a session that never paused must not start a fresh screen."""
    client = _client(monkeypatch, tmp_path, risk=0.2)
    resp = client.post(
        "/api/review/{sid}/decide".format(sid="review:test:not-waiting"),
        json={"answer": "131232131"},
    )
    assert resp.status_code == 409
