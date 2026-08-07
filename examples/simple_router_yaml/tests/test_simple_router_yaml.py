"""Offline tests for the ``simple_router_yaml`` example — no LLM, no network.

Loads ``workflow.yaml`` (the router as pure data) and runs it against a
mocked LLM transport to prove the supervisor loop works and terminates in
both directions: the happy path (the model says ``finish``) and the bounded
path (a model that never says ``finish`` cannot hang the graph).
"""

import json
import os

import httpx
import pytest

WORKFLOW_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflow.yaml"
)


def _load():
    from teff.yaml import load_workflow

    return load_workflow(WORKFLOW_PATH)


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

    def __init__(self):
        self.supervisor_calls = 0

    def _content_for(self, body: dict) -> str:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        if "router" in system:
            self.supervisor_calls += 1
            # the decider routes to an agent; the deterministic done guard
            # (done_keys={"code", "talk"}) then finishes the turn itself
            return "coder"
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


def test_yaml_validates():
    from teff.yaml_schema import validate_workflow_file

    assert validate_workflow_file(WORKFLOW_PATH) == []


def test_route_edges_exist():
    graph, _tools, _initial, _reducers = _load()
    edges = {(e.source_id, e.target_id, e.condition) for e in graph.edges}

    assert ("supervisor", "coder", "next_agent=coder") in edges
    assert ("supervisor", "talk", "next_agent=talk") in edges
    assert ("coder", "supervisor", None) in edges  # coder loops back
    assert ("talk", "supervisor", None) in edges  # talk loops back
    assert graph.entry_point == "supervisor"


@pytest.mark.asyncio
async def test_route_loop_runs_and_terminates(transport):
    graph, _tools, initial, reducers = _load()

    state = dict(initial)
    state["messages"] = [{"role": "user", "content": "list files with python"}]

    result = await graph.run(state, tools=[], reducers=reducers, max_iterations=80)

    # happy path: coder runs, then the done guard finishes deterministically —
    # a single supervisor LLM call, two supervisor rounds
    assert transport.supervisor_calls == 1
    assert result["code"] == "import os\nprint(os.listdir())"
    assert result["supervisor_rounds"] == 2

    # the assistant reply was appended to the shared conversation
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    assert any("print(os.listdir())" in m["content"] for m in assistant)


@pytest.mark.asyncio
async def test_bounded_loop_terminates_when_never_finish(monkeypatch):
    """A model that never says 'finish' must not hang the graph."""
    mock = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    graph, _tools, initial, reducers = _load()

    state = dict(initial)
    state["messages"] = [{"role": "user", "content": "list files with python"}]

    result = await graph.run(state, tools=[], reducers=reducers, max_iterations=100)

    # the done guard finishes once the agent has answered, without a hang
    assert result["supervisor_rounds"] <= 6
    assert mock.supervisor_calls < 100
