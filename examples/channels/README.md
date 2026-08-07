# Channels: one workflow.yaml, many transports

The same `workflow.yaml` runs over HTTP/SSE, a generic JSON webhook, a
Telegram bot, and plain terminal chat.  The workflow YAML — its `steps`,
`state`, `checkpoint:` and `channels:` blocks — is the single source of
truth.  No application code is needed for any transport; each adapter binds
the same durable `teff.channels.Assistant`, so checkpoints, message history
and interrupts behave identically everywhere.

Requires Ollama running locally (`ollama pull llama3.1:8b`).

## Run

```bash
# HTTP/SSE server (the channels.server block; --host/--port override)
teff serve examples/channels/workflow.yaml --port 8000

# Telegram bot (long-polling; channels.telegram block)
TELEGRAM_BOT_TOKEN=... teff bot examples/channels/workflow.yaml

# Telegram bot, webhook transport (channels.telegram.url must be set)
teff bot examples/channels/workflow.yaml --mode webhook

# Terminal chat — same workflow as a plain REPL conversation
teff chat examples/channels/workflow.yaml
```

## Terminal chat

`teff chat` builds the same durable Assistant and runs a REPL: each line is
one turn, the reply is printed, and a paused workflow (interrupt) asks
in-chat and resumes on your answer.  Ctrl-D or Ctrl-C exits.

```text
$ teff chat examples/channels/workflow.yaml
teff chat: session=chat-default owner=default (Ctrl-D to exit)
> The quick brown fox jumps over the lazy dog.
Быстрая коричневая лиса перепрыгивает через ленивую собаку.
```

## HTTP chat

```bash
curl -X POST localhost:8000/api/chat \
  -H 'X-User-Id: alice' -H 'Content-Type: application/json' \
  -d '{"message": "The quick brown fox jumps over the lazy dog."}'
# {"session_id": "...", "waiting": false, "reply": "..."}
```

`POST /api/chat/stream` returns the same turn as an SSE stream (token
events + a final `message` event with the full reply).  `GET /api/runs/{id}`
and `DELETE /api/runs/{id}` inspect or clear a conversation.  An interrupt
pauses the turn (`"waiting": true` + `prompt`/`key`); sending the operator's
answer as the next `message` resumes it.

## Webhook

```bash
curl -X POST localhost:8000/hooks/summarize \
  -H 'Content-Type: application/json' \
  -H 'X-User-Id: alice' \
  -d '{"text": "The quick brown fox jumps over the lazy dog."}'
# {"ok": true, "session_id": "...", "message": "..."}
```

The payload shape is declared as a JSON Schema in `channels.webhook`,
`input.message` templates it into the turn's message, and `session_key`
(here `text`) picks the durable session id.  `owner: "header.X-User-Id"`
scopes the checkpoints to the caller, so the same `text` from two users
stays in two isolated conversations.

## Telegram

Start the bot, then message it in Telegram: each chat is its own durable
session and each user owns their checkpoints, so a multi-turn workflow
pauses, asks in-chat, and resumes on your answer.

## Multi-agent supervisor over every channel

`supervisor/workflow.yaml` wraps the [supervisor
example](../supervisor_complex/) — planner + coder + a Map/loop quality gate
with an interrupt — in the same `channels:` block, so the whole team is
reachable over HTTP, webhook, Telegram and terminal from one file:

```bash
teff chat  examples/channels/supervisor/workflow.yaml
teff serve examples/channels/supervisor/workflow.yaml --port 8000
TELEGRAM_BOT_TOKEN=... teff bot examples/channels/supervisor/workflow.yaml
```

Ask it (in Russian) to implement something; it plans, codes, reviews and,
when the quality gate pauses, asks you in-channel:

```text
$ teff chat examples/channels/supervisor/workflow.yaml
> Реализуй функцию подсчёта суммы покупок.
Отправить код в работу? (approve / отлично / ok — да; любое другое — на доработку)
> отлично
```

Its webhook route is `POST /hooks/implement` with `{"task": "..."}`; the
payload's `task` becomes both the session id and the turn's message, and
`X-User-Id` scopes the checkpoints per caller.
