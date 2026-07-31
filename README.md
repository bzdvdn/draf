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

## Prompt templates

LLM nodes read *multiple* state keys into one prompt with `{key}` templates
(also supported in `system`)::

```python
node = LLM(
    model="llama3.1:8b",
    system="Ты инженер по ремонту.",
    prompt="Составь план для ремонта {type} на сумму {summ} рублей.",
    output_key="plan",
)
# state {"type": "кухни", "summ": 150000} -> user message:
# "Составь план для ремонта кухни на сумму 150000 рублей."
```

Values are stringified; a placeholder referencing a missing state key raises
`KeyError`. The underlying helper is `draf.prompt.render_template`.

## Dynamic fan-out (Map)

`Flow.map()` fans a state *list* out into parallel branches at runtime —
branch count is derived from the data, not declared up front. The processor
reads the same keys the Map fans out, so no glue node is needed::

```python
flow = Flow("repair-plans").map(
    LLM(
        model="llama3.1:8b",
        prompt="Составь план для ремонта {type} на сумму {summ} рублей.",
        output_key="plan",
    ),
    input_keys=["type", "summ"],   # lists zipped per index
    output_key="plans",            # list of per-item results
    max_concurrency=2,
)
result = await flow.compile().run(state={
    "type": ["кухни", "санузел"], "summ": [150000, 80000],
})
# -> {"plans": ["план для кухни...", "план для санузла..."]}
```

`chunk_size` batches items per branch, `max_concurrency` caps simultaneous
branches, and `result_key` overrides which per-item key to collect. Full demo:
`examples/map_repair_plans/` (Python with typed `State`, plus the same
workflow as YAML for `draf -f workflow.yaml`).

## Human-in-the-loop (interrupts)

Pause a workflow for operator input with an `Interrupt` node.  When
execution reaches it, `graph.run()` raises `GraphInterrupt`; resume with
the same `checkpoint_id` plus a `resume` dict::

```python
from draf.checkpoint import JSONFileCheckpointer
from draf.node.interrupt import GraphInterrupt
from draf.flow import Flow

flow = Flow("approval")
flow.step(LLM(model="llama3.1:8b", prompt="Составь план: {task}", output_key="draft"))
flow.interrupt(key="approved", prompt="Одобрить? (да / правки)")
flow.step(LLM(model="llama3.1:8b", prompt="{draft}\nВердикт: {approved}", output_key="final"))

graph = flow.compile()
cp = JSONFileCheckpointer("checkpoints")

try:
    await graph.run(state=state, checkpointer=cp, checkpoint_id="run-1")
except GraphInterrupt as interrupt:
    print(interrupt.prompt)          # "Одобрить? (да / правки)"
    answer = input("> ")
    result = await graph.run(
        state=state, checkpointer=cp,
        checkpoint_id="run-1", resume={"approved": answer},
    )
```

The answer lands in `state["approved"]` and execution continues past the
interrupt.  Interrupts require a `checkpointer`; the resume value for an
already-paused run is requested again if `resume` is missing.

### Revision loop

To re-ask on rejection, wire a cycle with `Flow.loop()` — a conditional
edge on the answer that sends execution back to the `Interrupt` node::

```python
flow.step(LLM(model="llama3.1:8b", prompt="Составь план: {task}", output_key="draft"))
flow.interrupt(key="approved", prompt="Одобрить? (да / правки)")
flow.loop(
    key="approved",
    until="да",
    done=LLM(model="llama3.1:8b", prompt="{draft}", output_key="final"),
    body=LLM(model="llama3.1:8b",
             prompt="Переработай {draft} с учётом: {approved}", output_key="draft"),
)
```

`loop()` wires `decider --key=until--> done` (stop) and
`decider --key!=until--> body -> decider` (repeat), so the graph returns
to the same `Interrupt` after each edit.  `max_iterations` caps the
rounds.  The decider can be any node that writes `key`, not just an
`Interrupt` — `loop()` also works for pure LLM self-check loops.

The same loop is described declaratively in YAML — conditional edges
already express the cycle, and `interrupt` is a registered node type.
See `examples/human_in_loop/workflow.yaml` for the full workflow;
running it still needs a `checkpointer` and a `resume` loop in Python.

## Streaming execution

`graph.stream()` runs the same execution core as `graph.run()` but yields
a `StreamEvent` for every observable step, so callers can render tokens and
progress before the run finishes.  Build the graph with the `Flow` API (or
directly with `Graph`) and stream it:

```python
from draf import Flow, LLM

flow = Flow("chat")
flow.step(LLM(model="llama3.1:8b", prompt="Скажи привет", output_key="answer"))
graph = flow.compile()

async for event in graph.stream(state):
    if event.type == "token":
        print(event.data["token"], end="", flush=True)
    elif event.type == "run_end":
        print("\nstatus:", event.data["status"])
```

Event types: `run_start`, `node_start`, `node_end`, `node_error`, `edge`,
`token`, `llm`, `interrupt`, `interrupt_resume`, `checkpoint`, `run_end`.
LLM tokens are emitted as they arrive (any node without tool calls streams
automatically in this mode); routing decisions, checkpoints, and interrupt
pauses are streamed the same way.  `stream()` accepts the same parameters as
`run()` — tools, checkpointer, resume, tracer, `max_iterations`.

See [streaming](examples/streaming/) — `run.py` (Flow) and `graph.py`
(low-level `Graph`) — for a full console demo.

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
| [map_repair_plans](examples/map_repair_plans/) | Dynamic fan-out (`Map`) + `{key}` prompt templates + typed `State` |
| [human_in_loop](examples/human_in_loop/) | Approve/Edit LLM output via `Interrupt` + `loop()` + resume (Python and YAML) |
| [react_agent](examples/react_agent/) | ReAct agent loop |
| [streaming](examples/streaming/) | Live LLM tokens + graph events via `graph.stream()` |
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
