"""Offline daemon tests — no LLM, no network.

Tests the file-backed job queue and the worker's drain step against a mocked
LLM transport: enqueue a job, process it, and assert the result is written
and the pending job removed.  Also proves a failing turn is recorded as an
error result rather than crashing the worker.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daemon import _process_job  # noqa: E402
from src.queue import enqueue, load_job, pending  # noqa: E402


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
def dirs(tmp_path):
    return {
        "queue_dir": str(tmp_path / "queue"),
        "results_dir": str(tmp_path / "results"),
    }


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    transport = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", transport)
    monkeypatch.setattr(httpx.AsyncClient, "stream", transport)
    monkeypatch.setenv("DRAF_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    from daemon import _build_assistant

    container, _ = _build_assistant()
    return container.assistant


def test_enqueue_and_pending(dirs):
    job_id = enqueue("help me", session_id="sess-1", queue_dir=dirs["queue_dir"])
    assert pending(queue_dir=dirs["queue_dir"]) == [job_id]
    job = load_job(job_id, queue_dir=dirs["queue_dir"])
    assert job["session_id"] == "sess-1"
    assert job["message"] == "help me"


def test_process_job_writes_result_and_removes_job(dirs, assistant):
    job_id = enqueue("help me", session_id="sess-1", queue_dir=dirs["queue_dir"])
    result = await_process(dirs, assistant, job_id)

    assert result["ok"] is True
    assert "1. Research." in result["result"]["plan"]
    assert pending(queue_dir=dirs["queue_dir"]) == []
    saved = json.loads(
        (Path(dirs["results_dir"]) / f"{job_id}.json").read_text("utf-8")
    )
    assert saved["ok"] is True


def test_supervisor_loop_is_bounded(tmp_path, monkeypatch):
    """A model that never says 'finish' must not hang the daemon graph."""
    import asyncio

    from src.graphs.build import build_flow
    from src.graphs.state import STATE_REDUCERS, initial_state

    transport = _MockTransport(finish=False)
    monkeypatch.setattr(httpx.AsyncClient, "post", transport)
    monkeypatch.setattr(httpx.AsyncClient, "stream", transport)

    flow, tools = build_flow()
    graph = flow.compile()
    state = initial_state()
    state["messages"] = [{"role": "user", "content": "help me"}]

    result = asyncio.run(
        graph.run(state, tools=tools, reducers=STATE_REDUCERS, max_iterations=100)
    )
    # the loop budget forces finish without a 100-iteration hang
    assert result["supervisor_rounds"] <= 6
    assert transport.supervisor_calls < 100


def test_process_failed_turn_records_error(dirs, monkeypatch):
    job_id = enqueue("boom", session_id="sess-2", queue_dir=dirs["queue_dir"])

    class _Exploding:
        async def run(self, *a, **k):
            raise RuntimeError("model exploded")

    result = await_process(dirs, _Exploding(), job_id)
    assert result["ok"] is False
    assert "model exploded" in result["error"]
    assert pending(queue_dir=dirs["queue_dir"]) == []


def await_process(dirs, assistant, job_id):
    import asyncio

    return asyncio.run(_process_job(assistant, job_id, **dirs))
