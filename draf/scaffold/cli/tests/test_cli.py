"""Offline CLI tests — no LLM, no network.

Runs the typer app against a mocked LLM transport to prove the ``run``
command drives one conversation turn through the supervisor flow and that
``chat`` is registered.  The mock serves both the non-streaming ``post``
path and the SSE ``stream`` path used by ReAct agents.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli import app  # noqa: E402


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
def runner(tmp_path, monkeypatch):
    transport = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", transport)
    monkeypatch.setattr(httpx.AsyncClient, "stream", transport)
    monkeypatch.setenv("DRAF_CHECKPOINT_DIR", str(tmp_path))
    return CliRunner()


def test_run_command_streams_one_turn(runner):
    result = runner.invoke(app, ["run", "help me"])
    assert result.exit_code == 0, result.output
    assert "-- run --" in result.output
    assert "-- supervisor_1 [supervisor] --" in result.output
    assert "1. Research. 2. Draft. 3. Review." in result.output
    assert "run_end: ok" in result.output


def test_run_command_accepts_session(runner):
    result = runner.invoke(app, ["run", "--session", "abc123", "help me"])
    assert result.exit_code == 0, result.output
    assert "session: abc123" in result.output


def test_chat_command_registered(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output
    assert "run" in result.output
