# Service Desk — the default `supervisor()` chat router

A support-desk app that shows the *configuration-driven* supervisor: a single
[`teff.node.Supervisor`](https://opencode.ai) node, added with
`Flow.supervisor()` and tuned entirely with config (no subclass), dispatches
every user message to one specialist agent.  This is the counterpart to
[`examples/applications/repair-ai-chat`](../repair-ai-chat), which shows the
alternative "one coordinator ReAct agent, experts as tools" design.

## What it demonstrates

* **One-word routing** — the supervisor LLM replies `billing`, `incident`,
  `deploy`, `fallback` or `finish`; `Flow.route()` sends the message to the
  matching specialist chain and loops control back to the supervisor.
* **`done_keys` guard** — once *any* specialist wrote its slot, the turn
  finishes deterministically **without a second LLM call** (one supervisor
  call per turn).
* **`fallback_agent`** — a premature `finish` on an empty turn routes to the
  fallback specialist instead of ending silently.
* **`max_rounds` budget** — `supervisor_rounds` bounds the loop, so even a
  model that never says `finish` cannot hang the graph.
* **Human-in-the-loop gateway** — the `deploy` route ends with an
  `Interrupt`; the run pauses, the operator's answer lands in
  `deploy_approved`, and the final summary honours it.
* **Durable multi-turn chat** — an entry `ContextBuilder` resets the
  per-turn scratch, so a follow-up message routes to a fresh specialist
  while `messages` keeps the whole conversation.
* **Domain-scoped knowledge base** — each specialist is bound to a dedicated
  RAG tool (`search_incident_knowledge` / `search_billing_knowledge` /
  `search_deploy_knowledge`) fed from `data/knowledge/*.csv`, so answers are
  grounded per domain instead of relying on the model's memory.

## Knowledge base

Seed documents live in `data/knowledge/{incidents,billing,deploy}.csv`
(Russian, one fact + `text` column per row, tagged by domain).  A
[`KnowledgeBase`](service_desk/rag/knowledge.py) lazily embeds and indexes them into a
store; each specialist is wired to exactly one `search_*_knowledge` tool via
`use_tools`.  Populate the store once before chatting:

```
# build (or rebuild) the store from the CSVs, then leave the app to load it
uv run python examples/applications/service_desk/cli.py load --rebuild --provider ollama
```

* `load` embeds into the default `data/knowledge/kb.db`.
* `--rebuild` clears the store first; otherwise `load` only ingests rows the
  store does not have yet.
* `--provider ollama` sets the embedding backend (see `service_desk/config/config.py`).

If the app is started without seeding, the first specialist search ingests
the CSVs on demand against the default store.

## The graph

```
reset (ContextBuilder) ─► supervisor ─ next_agent=billing ─► [ReAct billing] ─┐
                              ▲                                                  │
                              └──────────────────────────────────────────────────┘
        next_agent=deploy   → [ReAct deploy] → Interrupt(approval) → supervisor
        next_agent=incident → [ReAct incident] → supervisor
        next_agent=fallback → [ReAct fallback]  → supervisor
        next_agent=finish   → final LLM (ends the turn)
```

## Layout

```
service_desk/
├── cli.py               # interactive chat / one-shot turn (handles the gateway)
├── main.py                # uvicorn entry (factory lives in service_desk/server.py)
├── service_desk/
│   ├── api/              # FastAPI routers: chat (REST + SSE), runs, auth, health
│   ├── config/           # env-driven settings (.env / TEFF_* vars)
│   ├── graphs/           # state, prompts, supervisor flow builder
│   ├── rag/              # KnowledgeBase (embed + index + domain-scoped search)
│   ├── tools/            # search_*_knowledge tools bound to the KB
│   ├── core/             # dependency wiring (embedder, store, KB)
│   └── storage/          # JSON-file checkpointer + transient keys
├── tests/                # offline tests (no LLM, no network)
└── data/knowledge/       # CSV fact sources (incidents / billing / deploy)
```

## Run

Run from the repository root (the repo's `uv` project resolves `teff`).
Requires Ollama running locally for a live turn:

```
# 1. offline tests — wiring, routing, guards, gateway, multi-turn
uv run pytest examples/applications/service_desk/tests/

# 2. interactive chat (Ctrl-D/Ctrl-C to exit)
uv run python examples/applications/service_desk/cli.py

# 3. one-shot turn
uv run python examples/applications/service_desk/cli.py "сайт недоступен"
uv run python examples/applications/service_desk/cli.py "выкати изменения в прод"   # pauses for approval
```

If you want the specialists to ground answers in the knowledge base, first
`load` it (see above).  Without it, `chat` still works — the model just has
no lookup tool available.

## FastAPI server + observability

A FastAPI server wraps the same supervisor router and exposes chat (REST +
SSE), durable sessions, and a live trace dashboard — the observability half
stores full runs (topology, per-node spans, LLM payloads, token usage) in a
local SQLite store, browsable in the browser and via an API:

```
uv sync --extra fastapi
uv run python examples/applications/service_desk/main.py        # serves on 127.0.0.1:8000
```

Endpoints:

| Method | Path                      | Purpose                                        |
|--------|---------------------------|------------------------------------------------|
| POST   | `/api/chat`               | one-shot reply (`message`, optional `session_id`) |
| POST   | `/api/chat/stream`        | SSE event stream (tokens / routing / wait)     |
| GET    | `/api/runs/{id}`          | durable session state                          |
| DELETE | `/api/runs/{id}`          | delete a session                               |
| GET    | `/api/auth/verify`        | API-key probe                                  |
| GET    | `/api/health`             | server + model status                          |
| GET    | `/obs/ui`                 | trace dashboard (click on a run for the full span view) |
| GET    | `/obs/runs`               | recent runs (filter by status/name/owner/tag)  |
| GET    | `/obs/runs/{run_id}`      | full run record (topology, LLM calls, tools)   |

Example:

```
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
     -d '{"message": "выкати изменения в прод"}'     # pauses for approval, run_id returned
curl -s localhost:8000/obs/runs                       # list captured turns
```

Every chat turn is captured by a `GraphObserver` (`tracer` + full LLM payload
hook) and persisted via `SQLiteExporter`; the deploy gateway pauses the run as
an ordinary interrupt — the client answers the follow-up and resumes in the
same `session_id`. Sessions are durable (JSON-file checkpointer); restart the
server and a client can continue a conversation or revisit its traces.

Setup is env-driven via `TEFF_*` (see `service_desk/config/config.py`): model/provider,
`TEFF_API_KEY` (open to all when unset), `TEFF_TRACES_DB` / `TEFF_TRACES_PREFIX`,
`TEFF_HOST` / `TEFF_PORT`. Tests inject a stub knowledge base and in-memory
trace DB so the whole server is exercised without a network.

## Why the supervisor, not a ReAct coordinator

Contrast with `repair-ai-chat`: there, one coordinator agent drives the whole
pipeline and picks *experts as tools*; the sequence is emergent and lives in
the LLM's tool-calls.  Here the routing decision is a **single word with a
fixed vocabulary**, the expert slots are **visible state keys**, and the
guards (`done_keys`, `fallback_agent`, `route_keys`, `max_rounds`) make the
loop terminate and behave predictably without subclassing `Supervisor`.
Reach for this shape when you have a small set of discrete, self-contained
specialists and you want routing that is cheap, inspectable and safe.

## Add your own specialist

1. Add a prompt to `service_desk/graphs/prompts.py`.
2. Add an output slot to `ServiceDeskState` in `service_desk/graphs/state.py`.
3. Register the agent chain in `service_desk/graphs/build.py` under a new keyword in
   the `route(..., **agents)` call and — optionally — add the slot to
   `done_keys`.
4. Mention the new route value in the supervisor prompt.
