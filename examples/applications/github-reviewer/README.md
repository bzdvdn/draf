# GitHub reviewer daemon (multi-repo)

A production `workflow.yaml` that runs entirely through the `teff` CLI —
no per-app `daemon.py`.  The workflow **is** the daemon: each tick walks
every repository in `state.initial.repo_ids`, lists open pull requests,
reviews each new one, posts the verdict back to GitHub and notifies
Telegram.  Already-reviewed PRs are tracked in `kv_store` under
`reviewed-<repo>-<number>` so they are never re-reviewed.

## How it works

```
teff daemon -f workflow.yaml --interval 60
```

The CLI's `daemon` mode loads the workflow once, then re-runs it every
`--interval` seconds, carrying durable state between ticks (via the
optional `--checkpoint`).  One tick looks like:

```
reset (context_builder) ──► reviewer (react_agent) ──► tool_exec ──► (loop)
                                ▲                          │
                                └────────── _tool_call_name!= ┘
```

* `reset` — `context_builder` rebuilds the agent's `input` from the fresh
  `repo_ids` list and clears the previous tick's conversation, so each tick
  starts clean even when durable state is checkpointed.
* `reviewer` — a ReAct agent with all tools scoped in.  It drives the whole
  review loop itself: for every repo in `repo_ids` it lists open PRs with
  `github_list_open_prs`, fetches diffs with `github_get_pr_changes`,
  approves trivial PRs with `github_approve`, comments with
  `github_post_comment` otherwise, records each verdict in `kv_store`, and
  sends a Telegram summary.  Because the agent drives the loop, the same
  workflow handles any number of repositories.
* `tool_exec` — executes whatever tools the agent signalled and loops back
  until the agent stops calling tools.

## Requirements

* `teff` installed (this repo).
* A GitHub token with `pull_requests: write` permission (or a classic
  token with `repo` scope) for the reviewed repositories.
* A Telegram bot token and chat id.
* Ollama with `llama3.1:8b` (or point the workflow at any OpenAI-compatible
  provider via `react_agent` config).

## Configure

Env vars are interpolated into the workflow's tool configs (`${VAR}`).  Copy
`.env.example` and fill it in (or export the variables directly):

```bash
export GITHUB_TOKEN=github_pat_...
export TELEGRAM_BOT_TOKEN=123:abc
export TELEGRAM_CHAT_ID=-1001234567
```

Edit `workflow.yaml` and set the repositories to track in
`state.initial.repo_ids` (each is `owner/repo`):

```yaml
state:
  initial:
    repo_ids:
      - "acme/backend"
      - "acme/frontend"
```

## Run

```bash
# one tick (poll + review everything new), then exit
teff daemon -f workflow.yaml --once

# run forever, 60s between ticks
teff daemon -f workflow.yaml --interval 60
```

`kv_store` records already-reviewed PRs to `data/reviewed.json` and the
same tool instance lives across ticks, so dedup works without any
checkpoint.  Add `--checkpoint '{"type":"file","path":"data/daemon"}'` only
if you also want to persist counters and mid-review progress.

## Offline validation

```bash
teff validate workflow.yaml
```

## Tests

The core GitHub tools, the `tool_call` node and the multi-repo workflow
loading are covered offline in `tests/` (mocked HTTP transports, no
network).
