# repair-ai-chat — supervisor repair assistant (production scaffold app)

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
- **trace dashboard** — every chat turn is captured by a `GraphObserver`
  into `data/traces.db` and browsable at **`/obs/ui`** (one click per run:
  full graph, per-node LLM prompt/response, tags, notes); mounted via
  `draf.observability.attach_dashboard`, prefix from `DRAF_TRACES_PREFIX`
- production layout from `draf/scaffold`: `config/`, `src/`, `domain/`,
  `nodes/`, `tools/`, `rag/`, `graphs/`, `data/`

## Layout

```
repair-ai-chat/
├── main.py               # server entry point (uvicorn; host/port from settings)
├── app.py                # FastAPI app factory (uvicorn app:create_app)
├── cli.py                # interactive chat; or one repair-planning turn
├── src/                  # the production package
│   ├── config/           # env-driven settings (.env / DRAF_* vars)
│   ├── api/              # endpoint groups: router.py + chat/ + run/ + auth/
│   ├── core/             # dependency wiring (services, catalog)
│   ├── domain/           # entities + pure domain services (room/material/budget)
│   ├── graphs/           # typed state, prompts, JSON schemas, flow builder
│   ├── nodes/            # Extractor + context builders (Supervisor from draf.node)
│   ├── tools/            # Tool subclasses bound to services + catalog
│   ├── rag/              # materials catalog over an in-memory vector store
│   ├── service/          # Assistant: turn orchestration (HTTP + CLI)
│   └── storage/          # JSON-file checkpointer + session helpers
├── data/documents/       # materials.csv + price.csv — embedded lazily / via `load`
└── src/                  # (wiring + API tests live in tests/test_applications_repair_ai_chat.py)
```

## Run

Requires Ollama running locally:

```
ollama pull llama3.1:8b
uv run python examples/applications/repair-ai-chat/main.py
```

The end-to-end wiring test (`tests/test_applications_repair_ai_chat.py`)
runs the same graph against a mocked LLM transport, so it needs no network:
`uv run pytest tests/test_applications_repair_ai_chat.py`
