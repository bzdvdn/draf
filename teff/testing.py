"""Offline test doubles for LLM-backed nodes and the harness transport.

``teff.testing`` is a companion for writing fast, offline tests of
graphs that contain :class:`~teff.node.llm.LLM` or
:class:`~teff.node.agent.ReActAgent` nodes.  It provides two layers:

- :class:`FakeLLM` — a drop-in :class:`~teff.node.Node` subclass that
  answers with a canned string instead of calling a model.  Use it to
  build a graph whose model behaviour is fully deterministic.

- :func:`mock_llm` — a pytest fixture that patches
  :class:`~teff.harness.Harness` so *real* ``LLM`` / ``ReActAgent``
  nodes (including YAML-loaded workflows) run without network.  The
  fixture returns a :class:`MockLLM` record of every request.

No API keys or network access are required by either layer.
"""

from __future__ import annotations

import json
import typing

from teff.harness.loop import Harness
from teff.node.node import Node
from teff.prompt import render_template


class FakeLLM(Node):
    """An LLM node that returns a canned reply without any transport.

    Mirrors the ``LLM`` node contract for the common cases: ``system``
    and ``prompt`` templates are rendered from state, and the reply is
    written under ``output_key``.  ``content`` (default ``"mock"``)
    may itself be a ``{key}`` template.

    Examples:
        A graph with a single deterministic LLM step::

            from teff.testing import FakeLLM

            graph = Graph(
                nodes={"answer": FakeLLM({"prompt": "hi {name}", "content": "hello {name}"})},
                edges=[],
                entry_point="answer",
            )
    """

    type = "fake_llm"

    async def execute(self, ctx: typing.Any, state: dict) -> dict:
        cfg = self.config
        output_key = cfg.get("output_key", "output")
        prompt = cfg.get("prompt") or cfg.get("input_key") or ""
        if prompt:
            _ = render_template(str(prompt), state)  # validate placeholders
        content = render_template(str(cfg.get("content", "mock")), state)
        return {output_key: content}


class MockLLM:
    """Canned transport recording every model request.

    Instances expose ``content`` (the text every reply carries) and
    ``tool_calls`` (optional structured tool calls to attach), plus
    ``calls`` — the list of request bodies ``Harness`` built.  This
    lets assertions inspect prompts, model names, and tool schemas sent
    to the "model".
    """

    def __init__(self, content: str = "mock", tool_calls: list[dict] | None = None):
        self.content = content
        self.tool_calls = list(tool_calls or [])
        self.calls: list[dict] = []

    async def _post(self, body: dict) -> tuple[dict, bool]:
        self.calls.append(dict(body))
        msg: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        # Both OpenAI (``choices[0].message``) and Ollama (root ``message``)
        # response shapes, so the mock works for every provider type.
        return (
            {
                "choices": [{"message": msg}],
                "message": msg,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            False,
        )

    async def _post_stream(self, body: dict) -> tuple[str, dict]:
        self.calls.append(dict(body))
        return self.content, {"prompt_tokens": 10, "completion_tokens": 5}


try:
    import pytest
except ImportError:  # pragma: no cover - only when pytest is absent
    pytest = None  # type: ignore[assignment]

if pytest is not None:

    @pytest.fixture
    def mock_llm(monkeypatch):
        """Run ``LLM`` / ``ReActAgent`` nodes offline with canned replies.

        Patches :class:`~teff.harness.Harness` transport so any graph —
        built programmatically or loaded from YAML — never touches the
        network.  Returns a :class:`MockLLM` with ``content`` and
        ``tool_calls`` you can adjust mid-test and ``calls`` for
        assertions on the request bodies.

        Example::

            async def test_flow(mock_llm):
                mock_llm.content = "42"
                graph = ...
                result = await graph.run(state={...})
        """
        mock = MockLLM()
        monkeypatch.setattr(Harness, "_post", mock._post)
        monkeypatch.setattr(Harness, "_post_stream", mock._post_stream)
        return mock


def canned_json(value: dict) -> str:
    """Return *value* formatted as a JSON string an LLM might output."""
    return json.dumps(value, ensure_ascii=False)


__all__ = ["FakeLLM", "MockLLM", "canned_json"]
if pytest is not None:
    __all__ += ["mock_llm"]
