# FastAPI server for DRAFTFLOW graphs

A minimal production-style HTTP server around code-first graphs.  No
`workflow.yaml` anywhere — every graph is built in Python with the `Flow`
builder and registered by name in `graphs.py`.

Highlights:

- **Code-first graph registry** — add a graph by writing a function that
  returns a compiled `Flow` and registering it in `GRAPHS`.
- **Durable conversations** — each `chat_id` is a checkpoint in a
  per-user namespace.  Crashes, restarts and multi-turn messages all work
  off the same checkpointer; checkpoints are the conversation store.
- **Tenant isolation** — the `X-User-Id` header maps to the checkpoint
  `owner`, so the same `chat_id` under two users never collides.
- **SSE streaming** — `POST /api/chat/stream` forwards every
  `StreamEvent` (`token`, `node_start`, `checkpoint`, `run_end`, …) as a
  Server-Sent Event, so clients can render tokens live.
- **Tools and agents** — the `calculator` graph runs a ReAct loop
  (`react_agent` ↔ `tool_exec`) against an actual tool.

## Requirements

- Python 3.11+
- Ollama running locally with the model: `ollama pull llama3.1:8b`

## Install

```bash
uv sync --extra fastapi
```

## Run

```bash
uv run uvicorn --app-dir examples/fastapi_server app:app --port 8000
```

Checkpoints land in `examples/fastapi_server/data/checkpoints/<user>/`.
Override the location with the `TEFF_CHAT_CHECKPOINT_DIR` env var.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Server status + active model |
| `GET` | `/api/graphs` | List registered graphs and their metadata |
| `POST` | `/api/chat` | Run a graph (new chat, continuation, or interrupt resume) |
| `POST` | `/api/chat/stream` | Same, streamed as SSE events |
| `GET` | `/api/runs/{chat_id}` | Read a conversation's durable state |
| `DELETE` | `/api/runs/{chat_id}` | Delete a conversation |

Every request may carry an `X-User-Id` header that scopes the
conversation to that user (defaults to `default`).

### POST /api/chat

```jsonc
{
  "graph": "chat",            // one of: chat | calculator | summarize
  "message": "What is 17 * 24?",
  "chat_id": null,            // omit for a new conversation
  "max_iterations": 20
}
```

A new request without `chat_id` returns one; send it back to continue
the same conversation:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' -H 'X-User-Id: alice' \
  -d '{"graph":"chat","message":"Say hello"}'
```

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' -H 'X-User-Id: alice' \
  -d '{"graph":"chat","message":"What did I just ask?","chat_id":"<chat_id>"}'
```

When a graph pauses at an `Interrupt` node the response has
`"status": "interrupt"` with a `key` and `prompt`; resume it with a
`resume` map instead of a new message:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' -H 'X-User-Id: alice' \
  -d '{"graph":"chat","chat_id":"<chat_id>","resume":{"approved":"да"}}'
```

### POST /api/chat/stream

Same body, but the reply is an SSE stream.  Each event is named after the
`StreamEvent.type` and carries the event payload plus `chat_id`:

```
event: token
data: {"chat_id":"...","token":"1","provider":"ollama","model":"llama3.1:8b"}
```

The first event is always `chat_id`, so the client learns the durable id
before generation finishes.

## How it works

- `graphs.py` — `build_chat()`, `build_calculator()`, `build_summarize()`
  each define a graph with the `Flow` builder (`react()`, `llm()`, ...)
  and are collected in the `GRAPHS` registry.
- `app.py` — routes pick a graph by name, resolve the owner from
  `X-User-Id`, and run it with a shared `JSONFileCheckpointer`.  Multi-turn
  continuation loads the checkpoint, appends the new user message to
  `messages`, re-enters the graph at its entry point, and lets the graph
  finish (see `_prepare_turn`).
