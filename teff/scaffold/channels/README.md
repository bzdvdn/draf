# {{PROJECT_NAME}}

A YAML-first teff app: one `workflow.yaml` is the single source of truth,
reachable over every transport with **no application code**.  The
`channels:` block declares the adapters; `teff serve` / `teff bot` run them
against the same compiled `Assistant`, so checkpoints, message history and
interrupts behave identically everywhere.

Requires Ollama locally (`ollama pull llama3.1:8b`).

## Run

```bash
uv sync

# HTTP/SSE server (+ the /hooks/summarize webhook)
teff serve workflow.yaml --port 8000

# Telegram bot (long-polling; channels.telegram block)
TELEGRAM_BOT_TOKEN=... teff bot workflow.yaml

# Telegram webhook transport (set channels.telegram.url)
teff bot workflow.yaml --mode webhook
```

## HTTP chat

```bash
curl -X POST localhost:8000/api/chat \
  -H 'X-User-Id: alice' -H 'Content-Type: application/json' \
  -d '{"message": "The quick brown fox jumps over the lazy dog."}'
# {"session_id": "...", "waiting": false, "message": "..."}
```

`POST /api/chat/stream` streams the same turn as SSE.  An interrupt pauses
with `"waiting": true` + `message` (the prompt); sending the operator's
answer as the next `message` resumes it.

## Webhook

```bash
curl -X POST localhost:8000/hooks/summarize \
  -H 'Content-Type: application/json' \
  -d '{"text": "The quick brown fox jumps over the lazy dog."}'
# {"ok": true, "session_id": "...", "message": "..."}
```

## HOW TO EXTEND

* Add a step to `steps:` — every node type in `teff` is available.
* Add a `channels.webhook` entry (any path, JSON Schema, `input.message`
  template, `session_key`) to expose a new inbound hook.
* Change the checkpoint backend under `checkpoint:`.
* Set `channels.server.host`/`port` or `channels.telegram` as needed.
