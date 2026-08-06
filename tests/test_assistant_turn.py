"""Tests for the abstract ``Assistant.turn`` / ``pending`` / ``stream`` API.

Builds the ``repair-ai-chat`` graph (a two-approval human-in-the-loop flow)
and drives it through the unified ``Assistant.turn`` loop, proving that
``GraphInterrupt`` pauses are folded into a ``TurnResult`` (``waiting``)
instead of leaking to the caller, and that ``Assistant`` resumes the run
transparently from the durable checkpoint.
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
from src.storage import TRANSIENT_KEYS, build_checkpointer  # noqa: E402

from draf import Assistant, TurnResult  # noqa: E402


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
    """Canned, system-prompt-aware LLM transport (see the app wiring test)."""

    def __init__(self):
        self.calls: list[str] = []

    def _content_for(self, body: dict) -> str:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        self.calls.append(system[:40])
        if "Supervisor" in system:
            return "pipeline"
        if "извлекаешь" in system:
            return json.dumps({"room_type": "bathroom", "area": 5.0})
        if "классифицируешь" in system:
            user = " ".join(
                m.get("content", "")
                for m in body.get("messages", [])
                if m.get("role") == "user"
            )
            return json.dumps(
                {"ok": any(w in user.lower() for w in ("да", "ок", "конечно")), "message": ""}
            )
        if "Planner" in system:
            return "1. Демонтаж. 2. Стены. 3. Пол. 4. Отделка."
        if "Estimator" in system:
            return "Смета: стены 12000."
        if "Materials Agent" in system:
            return "Плитка 780 ₽/м²."
        if "QA Agent" in system:
            return json.dumps({"ok": True, "message": ""})
        return "Помогу спланировать ремонт."

    def __call__(self, *args, **kwargs):
        content = self._content_for(kwargs.get("json") or {})
        if args and args[0] == "POST":

            class _Resp:
                def raise_for_status(self):
                    pass

                async def aiter_lines(self):
                    for line in _stream_lines(content):
                        yield line

            class _CM:
                async def __aenter__(self):
                    return _Resp()

                async def __aexit__(self, *exc):
                    return False

            return _CM()

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


def _build_assistant(checkpoint_dir: str) -> Assistant:
    flow, tools = build_flow()
    return Assistant(
        flow.compile(),
        tools,
        build_checkpointer(checkpoint_dir),
        reducers=STATE_REDUCERS,
        initial_state=initial_state,
        transient_keys=TRANSIENT_KEYS,
    )


@pytest.mark.asyncio
async def test_turn_loop_handles_interrupts(transport, tmp_path):
    """turn() folds each pause into waiting=True and resumes transparently."""
    assistant = _build_assistant(str(tmp_path))
    sid = "sess-turn"

    first = await assistant.turn(sid, "Спланируй ремонт ванной 5 м².")
    assert isinstance(first, TurnResult)
    assert first.waiting is True
    assert first.prompt  # the plan-approval question
    assert first.session_id == sid
    assert await assistant.pending(sid) is not None  # durable pause is visible

    # "да" approves the plan; the run continues to the estimate-approval pause
    second = await assistant.turn(sid, "да")
    assert second.waiting is True
    assert "Смета" in (second.prompt or "")

    # "да" approves the estimate; the pipeline completes
    done = await assistant.turn(sid, "да")
    assert done.waiting is False
    assert "Смета:" in done.reply
    assert done.key is None
    # after completion there is no longer a pending interrupt
    assert await assistant.pending(sid) is None


@pytest.mark.asyncio
async def test_turn_fresh_session_and_durable_history(transport, tmp_path):
    """A fresh session (no pending pause) starts from the entry point."""
    assistant = _build_assistant(str(tmp_path))
    sid = "sess-fresh"
    assert await assistant.pending(sid) is None
    result = await assistant.turn(sid, "Привет")
    # no interrupt raised at the entry router, so the turn runs to the end
    assert isinstance(result, TurnResult)


@pytest.mark.asyncio
async def test_stream_yields_interrupt_and_ends(transport, tmp_path):
    """stream() surfaces a terminal interrupt event instead of raising."""
    assistant = _build_assistant(str(tmp_path))
    sid = "sess-stream"

    types = []
    async for ev in assistant.stream(sid, "Спланируй ремонт ванной 5 м²."):
        types.append(ev.type)
        if ev.type == "interrupt":
            break

    assert "run_start" in types
    assert types[-1] == "interrupt"
    # the durable summary counts the plan-approval pause already present
    assert await assistant.pending(sid) is not None