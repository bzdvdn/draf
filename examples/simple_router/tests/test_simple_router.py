"""Offline tests for the ``simple_router`` example — no LLM, no network.

Runs the ``route()`` flow against a mocked LLM transport to prove the
supervisor loop terminates in both directions: the happy path (the model
says ``finish``) and the bounded path (a model that never says ``finish``
cannot hang the graph).
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.graphs.build import build_flow  # noqa: E402
from src.graphs.state import STATE_REDUCERS, initial_state  # noqa: E402


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
        if "router" in system:
            self.supervisor_calls += 1
            if self.finish:
                # alternate coder -> finish so multi-turn sessions route too
                return "coder" if self.supervisor_calls % 2 == 1 else "finish"
            return "coder"  # never finish — proves the loop budget
        if "coding assistant" in system:
            return "import os\nprint(os.listdir())"
        return "Hello! How can I help?"

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

    assert ("supervisor_1", "subflow_2", "next_agent=coder") in edges
    assert ("supervisor_1", "subflow_3", "next_agent=talk") in edges
    assert ("subflow_2", "supervisor_1", None) in edges  # coder loops back
    assert ("subflow_3", "supervisor_1", None) in edges  # talk loops back
    assert flow.compile().entry_point == "supervisor_1"


@pytest.mark.asyncio
async def test_route_loop_runs_and_terminates(transport):
    flow = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "list files with python"}]

    result = await graph.run(state, tools=[], reducers=STATE_REDUCERS, max_iterations=80)

    # happy path: coder -> finish, exactly two supervisor calls
    assert transport.supervisor_calls == 2
    assert result["code"] == "import os\nprint(os.listdir())"
    assert result["supervisor_rounds"] == 2

    # the assistant reply was appended to the shared conversation
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    assert any("print(os.listdir())" in m["content"] for m in assistant)


@pytest.mark.asyncio
async def test_bounded_loop_terminates_when_never_finish(monkeypatch):
    """A model that never says 'finish' must not hang the graph."""
    mock = _MockTransport(finish=False)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "list files with python"}]

    result = await graph.run(state, tools=[], reducers=STATE_REDUCERS, max_iterations=100)

    # the loop budget forces finish without a 100-iteration hang
    assert result["supervisor_rounds"] == 6
    assert mock.supervisor_calls < 100


@pytest.mark.asyncio
async def test_assistant_turn_is_durable(transport, tmp_path):
    """run_turn persists the conversation across a second turn."""
    from src.service.assistant import Assistant
    from src.storage import build_checkpointer

    flow = build_flow()
    assistant = Assistant(flow.compile(), build_checkpointer(tmp_path))

    first = await assistant.run_turn("sess-1", "list files with python")
    assert first["code"] == "import os\nprint(os.listdir())"

    second = await assistant.run_turn("sess-1", "now make it a class")
    # history is preserved: two user + two assistant messages
    users = [m for m in second["messages"] if m.get("role") == "user"]
    assert len(users) == 2
    assert transport.supervisor_calls >= 4
