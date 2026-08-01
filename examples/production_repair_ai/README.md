# production_repair_ai — supervisor repair assistant (production scaffold app)

A runnable instance of the **production scaffold template** (`draf/scaffold`):
a `src/` package with typed state, domain services, per-agent
tools, a RAG materials catalog — wired as a **supervisor Flow** built on
[`Flow.route()`](../../draf/flow/flow.py).

## Flow

```
supervisor ─ next_agent=direct ──► ContextBuilder → ReAct(direct) ─┐
    ▲                                                              │
    │  each agent runs as a SubFlow; control returns to the        │
    └────────────────────────────── supervisor ────────────────────┘

supervisor ─ next_agent=finish ──► Extractor (structured project_info)
```

The `Supervisor` LLM node writes `next_agent`; `route()` sends each value to
the matching agent chain and loops back to the supervisor. On `finish` the
route exits through the `Extractor`, which pulls structured project info
from the whole conversation.

## Highlights

- **`Flow.route("next_agent", finish=..., direct=..., planner=..., ...)`** —
  supervisor-style routing with a finish chain
- agent chains as **`SubFlow`** (context builder → ReAct harness with
  tool scoping → assistant append), each with a private scratch conversation
- **`graph.run(emit=...)`** — stream `StreamEvent`s while still returning
  the final state (nested `run_start`/`run_end` are stripped)
- production layout from `draf/scaffold`: `config/`, `src/`, `domain/`,
  `nodes/`, `tools/`, `rag/`, `graphs/`, `data/`

## Layout

```
production_repair_ai/
├── main.py               # server entry point (uvicorn; host/port from settings)
├── app.py                # FastAPI app factory (uvicorn app:create_app)
├── cli.py                # debug: one streaming repair-planning turn
├── src/                  # the production package
│   ├── config/           # env-driven settings (.env / DRAF_* vars)
│   ├── api/              # endpoint groups: router.py + chat/ + run/ + auth/
│   ├── core/             # dependency wiring (services, catalog)
│   ├── domain/           # entities + pure domain services (room/material/budget)
│   ├── graphs/           # typed state, prompts, JSON schemas, flow builder
│   ├── nodes/            # Supervisor, Extractor, context builders
│   ├── tools/            # Tool subclasses bound to services + catalog
│   ├── rag/              # materials catalog over an in-memory vector store
│   ├── service/          # Assistant: turn orchestration (HTTP + CLI)
│   └── storage/          # JSON-file checkpointer + session helpers
├── data/documents/       # materials.csv — embedded lazily on first search
└── src/                  # (wiring + API tests live in tests/test_examples_production_repair_ai.py)
```

## Run

Requires Ollama running locally:

```
ollama pull llama3.1:8b
uv run python examples/production_repair_ai/main.py
```

The end-to-end wiring test (`tests/test_examples_production_repair_ai.py`)
runs the same graph against a mocked LLM transport, so it needs no network:
`uv run pytest tests/test_examples_production_repair_ai.py`
