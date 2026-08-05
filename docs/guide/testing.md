# Testing offline

Every graph that talks to a model can be tested without keys or network via
`draf.testing`. It ships two layers:

- `FakeLLM` — a deterministic node you put *in place of* `LLM` in a
  programmatically-built graph.
- `mock_llm` — a pytest fixture that patches the harness transport so *real*
  `LLM` / `ReActAgent` nodes (including YAML-loaded workflows) answer with
  canned text.

## FakeLLM — deterministic graphs

Build a graph whose model step always returns the same string:

```python
import asyncio
from draf.graph import Graph
from draf.testing import FakeLLM

g = Graph(
    nodes={"answer": FakeLLM({"prompt": "hi {name}", "content": "hello {name}"})},
    edges=[],
    entry_point="answer",
)
result = asyncio.run(g.run(state={"name": "Ana"}))
assert result["output"] == "hello Ana"
```

`FakeLLM` renders `system` / `prompt` templates and writes the reply under
`output_key` (default `"output"`). `content` may itself use `{key}`
placeholders.

## mock_llm — real nodes, no network

The `mock_llm` fixture intercepts `Harness` so the real LLM node runs its
full pipeline (tool calling, structured output, streaming) against canned
responses. It returns a `MockLLM` with:

- `content` — the text every reply carries (adjust it mid-test).
- `tool_calls` — optional structured tool calls to attach.
- `calls` — the request bodies sent, for asserting on prompts/models.

```python
from draf.provider import ProviderRegistry


async def test_flow(mock_llm):
    mock_llm.content = "42"

    g = Graph(
        nodes={
            "a": LLM(
                {
                    "model": "gpt-4",
                    "prompt": "calc",
                    "output_key": "answer",
                    "provider": "openai",
                }
            )
        },
        edges=[],
        entry_point="a",
        providers=ProviderRegistry.from_presets("openai"),
    )
    result = await g.run(state={})
    assert result["answer"] == "42"
    assert mock_llm.calls[0]["model"] == "gpt-4"
```

Structured output works too — feed the fixture a valid JSON string via
`canned_json`:

```python
from draf.testing import canned_json

mock_llm.content = canned_json({"answer": 42, "ok": True})
```

`draf.testing` is registered as a `pytest11` entry point, so `mock_llm` is
available in any downstream test suite without a `conftest.py`.
