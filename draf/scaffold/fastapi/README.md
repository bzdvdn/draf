# {{PROJECT_NAME}}

A production draf app scaffolded with `draf new {{project_slug}}`, organized
as a **FastAPI service**: a supervisor Flow built on `Flow.route()` with
typed state, durable sessions, API-key auth, and a debug `cli.py`.  Keep
this skeleton generic and add your own agents, tools and state — see
**Add your own agent** below.  For a fully-worked real example, look at
`examples/applications/repair-ai-chat`.

## Layout

```
{{project_slug}}/
├── main.py               # server entry point (uvicorn; host/port from settings)
├── app.py                # FastAPI app factory (uvicorn app:create_app)
├── cli.py                # debug: one streaming turn against Ollama
├── src/                  # the production package
│   ├── config/           # env-driven settings (.env / DRAF_* vars)
│   ├── core/             # composition root: src/core/container.py build_container
│   ├── api/              # FastAPI endpoint groups
│   │   ├── router.py     #   api_router — aggregates chat + run + auth + health
│   │   ├── chat/         #   POST /api/chat, /api/chat/stream
│   │   ├── run/          #   GET/DELETE /api/runs/{chat_id}
│   │   └── auth/         #   X-API-Key dependency + /api/auth/verify
│   ├── graphs/           # state, prompts, supervisor flow builder
│   ├── nodes/            # Supervisor + context builders
│   ├── tools/            # Tool subclasses handed to the agents
│   ├── service/          # Assistant: turn orchestration (HTTP + CLI)
│   └── storage/          # JSON-file checkpointer + session helpers
├── data/checkpoints/     # durable session state (created at runtime, git-ignored)
└── tests/                # wiring + API tests (offline, no API keys)
```

## How the graph works

```
supervisor ─ next_agent=planner ──► ContextBuilder → ReAct ─┐
    ▲                                                       │
    └────────────────── supervisor ◀────────────────────────┘
   (next_agent=writer / reviewer)  ...  (next_agent=finish → exits)
```

The `Supervisor` LLM node writes `next_agent`; `route()` sends each value to
the matching agent chain (a `SubFlow`: context builder → ReAct harness with
tools → append the reply to the conversation) and loops back to the
supervisor. When it says `finish`, the loop exits.

## Configuration

Settings live in `src/config/config.py` and are read from the environment
(`DRAF_*` vars) or a local `.env` file:

```
DRAF_PROVIDER=ollama
DRAF_MODEL=llama3.1:8b
DRAF_API_KEY=secret            # empty = auth disabled
DRAF_HOST=127.0.0.1
DRAF_PORT=8000
```

## Add your own agent

Each piece carries a `HOW TO EXTEND` comment.  The usual loop:

1. Add a prompt to `src/graphs/prompts.py`.
2. Add an output slot to `AppState` in `src/graphs/state.py` (only if the
   agent produces shared state).
3. Build the agent chain in `src/graphs/build.py` with
   `agent_step(system, output_key, use_tools)` and register it under a new
   keyword in the `route(..., **agents)` call.
4. Mention the new route value in the supervisor prompt, and (if the agent
   uses tools) add a `Tool` subclass in `src/tools/` and register it in
   `build_tools()`.

## Run

```
# 1. install the package (draf must be importable)
uv sync

# 2. offline tests — wiring + API, no LLM required
uv run pytest tests/

# 3. debug a single turn against a local Ollama (streams tokens)
uv run python cli.py "Help me draft a note"

# 4. serve the API
uv run python main.py
```

## API

```
# health + list registered endpoints
curl http://localhost:8000/api/health

# single-shot reply
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-User-Id: alice" \
  -d '{"message": "Hi! Where should we start?"}'

# SSE stream (tokens as they arrive)
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-User-Id: alice" \
  -d '{"message": "Hi! Where should we start?"}'
```

Sessions are durable: send the same `session_id` (or let the server assign
one) to continue a conversation.

## Trace dashboard

Every chat turn is captured by a `GraphObserver` into `data/traces.db`
(SQLite) and browsable out of the box:

```
open http://localhost:8000/obs/ui
```

The dashboard lists runs (filters + pagination), opens each run on its own
page (full graph topology, per-node LLM prompt/response, copy buttons) and
supports editable tags and notes.  Pointers:

- `DRAF_TRACES_DB` — where the traces are stored (default `data/traces.db`)
- `DRAF_TRACES_PREFIX` — URL prefix for the dashboard (default `/obs`)
- mount it in your own app with
  `draf.observability.attach_dashboard(app, exporter, prefix=...)`

## Template placeholders

Rendered by `draf new`: `{{PROJECT_NAME}}`, `{{project_slug}}`,
`{{ProjectName}}`.
