"""Offline tests for the ``support_triage`` example — no LLM, no network.

Runs the ``route()`` flow against a mocked LLM transport to prove the
supervisor loop terminates in both directions and that the human escalation
**pauses** on the ``escalate`` interrupt and **resumes** from the checkpoint.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from support_triage.graphs.build import build_flow  # noqa: E402
from support_triage.graphs.state import STATE_REDUCERS, initial_state  # noqa: E402

from teff.node.interrupt import GraphInterrupt  # noqa: E402


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


class _MockTransport:
    """Serves a canned, system-prompt-aware reply per LLM call."""

    def __init__(self, *, escalate: bool = False):
        self.supervisor_calls = 0
        self.escalate = escalate

    def _content_for(self, body: dict) -> str:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        if "support router" in system:
            self.supervisor_calls += 1
            # the done guard (done_keys) finishes the turn deterministically
            return "human" if self.escalate else "billing"
        if "billing support assistant" in system:
            return (
                "По вашей квитанции: списание произошло вчера, вернём в течение 3 дней."
            )
        if "escalation note writer" in system:
            return "Клиент требует возврат за неоказанную услугу — нужен оператор."
        return "Проверьте версию приложения и обновитесь."

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


def test_route_edges_exist():
    flow = build_flow()
    edges = {(e.source_id, e.target_id, e.condition) for e in flow.compile().edges}
    conditions = {cond for _, _, cond in edges if cond}

    assert flow.compile().entry_point == "supervisor_1"
    assert "next_agent=billing" in conditions
    assert "next_agent=tech" in conditions
    assert "next_agent=human" in conditions
    # every sub-flow loops back to the supervisor
    assert any(target == "supervisor_1" and cond is None for _, target, cond in edges)


@pytest.mark.asyncio
async def test_billing_answers_and_terminates(transport):
    flow = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["ticket"] = "Почему сняли деньги вчера?"
    state["messages"] = [{"role": "user", "content": state["ticket"]}]

    result = await graph.run(
        state, tools=[], reducers=STATE_REDUCERS, max_iterations=80
    )

    # one supervisor call: route to billing, then the done guard finishes
    assert transport.supervisor_calls == 1
    assert "квитанци" in result["billing"]
    assert result["supervisor_rounds"] == 2

    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    assert any("квитанци" in m["content"] for m in assistant)


@pytest.mark.asyncio
async def test_bounded_loop_terminates_when_never_finish(monkeypatch):
    """A supervisor that never says 'finish' must not hang the graph."""
    mock = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow = build_flow()
    graph = flow.compile()
    state = initial_state()
    state["ticket"] = "Не работает вход."
    state["messages"] = [{"role": "user", "content": state["ticket"]}]

    result = await graph.run(
        state, tools=[], reducers=STATE_REDUCERS, max_iterations=100
    )

    assert result["supervisor_rounds"] <= 6
    assert mock.supervisor_calls < 100


@pytest.mark.asyncio
async def test_escalation_pauses_and_resumes(monkeypatch, tmp_path):
    """The human escalation pauses on the interrupt and resumes durably."""
    mock = _MockTransport(escalate=True)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    from teff.checkpoint import SQLiteCheckpointer

    flow = build_flow()
    graph = flow.compile()
    cp = SQLiteCheckpointer(str(tmp_path / "cp.db"))

    state = initial_state()
    state["ticket"] = "Верните деньги, это мошенничество!"
    state["messages"] = [{"role": "user", "content": state["ticket"]}]

    with pytest.raises(GraphInterrupt):
        await graph.run(
            state,
            checkpointer=cp,
            checkpoint_id="ticket-1",
            tools=[],
            reducers=STATE_REDUCERS,
            max_iterations=50,
        )

    result = await graph.run(
        state,
        checkpointer=cp,
        checkpoint_id="ticket-1",
        resume={"escalate": "передать старшему оператору"},
        tools=[],
        reducers=STATE_REDUCERS,
        max_iterations=50,
    )

    # the handover note was drafted before the pause, and the operator's
    # answer resumed the run to completion
    assert result["escalate"] == "передать старшему оператору"
    assert "оператор" in result["human_note"]

    # the note reached the shared conversation after resume
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    assert any("оператор" in m["content"] for m in assistant)
