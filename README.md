# Draf

**Workflow as data. Agents as graphs.**

Draf is a Python framework for building durable AI agents and workflows —
an embeddable async library. Inspired by LangGraph and LangChain, it brings
graph-based, stateful agents to Python with minimal dependencies and zero
runtime magic.

- **Async by default** — nodes, tools, and LLM calls are all `async`.
- **Workflow as data** — the canonical graph is YAML/JSON; code is optional.
- **Durable** — checkpoint/resume across file, SQLite, and PostgreSQL backends.
- **Embeddable** — a library, not a platform. You import us; we never import you.

## Install

```bash
pip install draf
# extras: draf[embedding] for RAG stores, draf[pg-checkpoint] for PostgreSQL checkpoints
```

Python >= 3.11. Core runtime depends only on `httpx`, `pyyaml`, and `typer`.

## Quick start

### YAML workflow

```yaml
name: text-pipeline
state:
  initial:
    title: "  hello world  "

steps:
  - id: trim
    type: transform
    config: {action: trim, input_key: title, output_key: trimmed}
  - id: uppercase
    type: transform
    config: {action: uppercase, input_key: trimmed, output_key: loud}
  - id: count
    type: transform
    config: {action: count_lines, input_key: loud, output_key: line_count}

edges:
  - from: trim
    to: uppercase
  - from: uppercase
    to: count
```

```python
import asyncio
from draf.yaml import load_workflow

async def main():
    graph, tools, state, reducers = load_workflow("workflow.yaml")
    result = await graph.run(state, tools=tools, reducers=reducers)
    print(result)

asyncio.run(main())
```

### Flow API (Python)

```python
import asyncio
from draf import set_defaults
from draf.flow import Flow, Case
from draf.node import LLM, Transform

set_defaults(provider="ollama")

async def main():
    flow = Flow("sentiment-router")
    flow.step(LLM(
        model="llama3.1:8b",
        system='Classify the sentiment. Reply "positive" or "negative".',
        input_key="text", output_key="sentiment",
    ))
    flow.branch(
        "sentiment",
        Case("positive").add(Transform(action="value", value="Glad you liked it!", output_key="reply")),
        Case("negative").add(Transform(action="value", value="Sorry to hear that.", output_key="reply")),
    ).converge(Transform(action="uppercase", input_key="reply", output_key="result"))

    result = await flow.compile().run(state={"text": "I love this product!"})
    print(result)

asyncio.run(main())
```

## Core concepts

- **State** — a flat, JSON-serializable dict. Nodes transform state, nothing else.
- **Nodes** — pure `async def fn(ctx, state) -> dict` functions (or built-ins:
  `LLM`, `Transform`, `ReActAgent`, `ToolExec`).
- **Graph** — nodes + edges, including conditional edges, branches, and
  `__error__` fallbacks. The graph owns routing and resilience.
- **Tools** — implement `Tool` or use the `@tool` decorator; shareable across nodes.
- **RAG** — `RAGTool` over pluggable vector stores (`InMemoryVectorStore`,
  `SQLite`, `Chroma`, `Qdrant`, `PGVector`).

## Durable execution (checkpoints)

`Graph.run()` accepts a `checkpointer` and a `checkpoint_id`. A checkpoint is
written **before** every node, so a crash or error resumes from the last safe
point instead of starting over.

```python
from draf import Graph
from draf.checkpoint import SQLiteCheckpointer
from draf.node import Transform

nodes = {"shout": Transform(action="uppercase", input_key="text", output_key="loud")}
graph = Graph(nodes, edges=[], entry_point="shout")
cp = SQLiteCheckpointer("checkpoints.db")

# first run crashes at some node
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")

# same id resumes from the saved checkpoint and completes
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")
```

Backends: `JSONFileCheckpointer` (core), `SQLiteCheckpointer` (core),
`PGCheckpointer` (`draf[pg-checkpoint]`, needs PostgreSQL). On resume the saved
state wins over the passed-in state; a `State` instance keeps its schema and
reducers.

## Parallel branches

Run independent branch chains concurrently and merge their results with
`Flow.parallel()` — each branch gets an isolated copy of the state, and
per-key reducers merge updates back so `append` branches accumulate instead
of overwriting each other.

```python
from draf.flow import Flow
from draf.node import Transform

flow = Flow("p").parallel(
    [Transform(action="uppercase", input_key="title", output_key="title")],
    [Transform(action="uppercase", input_key="body", output_key="body")],
).converge(Transform(action="value", value="done", output_key="status"))

result = await flow.compile().run(state={"title": "hi", "body": "world"})
# -> title/body uppercased in parallel, then status="done"
```

Branches can be single nodes, lists of nodes (run sequentially inside the
branch), or embedded `Flow` subgraphs. The parallel node also works directly:
`Parallel([[node1], [node2]])`.

For a full end-to-end demo with LLM calls, see
`examples/parallel/rag_report.py` — two RAG searches run in parallel branches,
an LLM merges the summaries into a report file, and a final LLM reviews it
(`VERDICT: pass/fail`). Requires local Ollama.

## Observability (telemetry)

Pass a `RunTracer` to `graph.run()` to collect a JSON-serialisable event log:
node start/end with latency, edge routing, checkpoints, retries, and LLM token
usage. Fold it into a summary afterwards.

```python
from draf import Graph, RunTracer

tracer = RunTracer()
await graph.run(state, tracer=tracer)

print(tracer.to_json())        # {"summary": {...}, "events": [...]}
print(tracer.summary())        # RunSummary(status, total_ms, nodes, tokens, ...)
```

The CLI exposes the same report: `draf -f workflow.yaml --trace`.

## Examples

| Example | What it shows |
| ------- | ------------- |
| [basic_pipeline](examples/basic_pipeline/) | Minimal YAML pipeline, no API keys |
| [branching](examples/branching/) | Conditional edges + Flow API |
| [parallel](examples/parallel/) | Concurrent branches + typed `State` reducers |
| [react_agent](examples/react_agent/) | ReAct agent loop |
| [rag_search](examples/rag_search/) | RAG over a local CSV, in-memory store |
| [rag_stores](examples/rag_stores/) | Same RAG agent on every vector store |
| [checkpoint_resume](examples/checkpoint_resume/) | Crash/resume in a few lines |
| [checkpoint_stores](examples/checkpoint_stores/) | Durable workflow on file/sqlite/pg |

All LLM examples run on local Ollama (`ollama pull llama3.1:8b`) — no API keys.

## Development

```bash
uv sync                        # install deps
uv run pytest tests/ -q        # tests
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run mypy .                  # types
```

See [CONSTITUTION.md](CONSTITUTION.md) for the framework's principles and
non-negotiable rules.
