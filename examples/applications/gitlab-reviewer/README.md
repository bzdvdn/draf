# GitLab reviewer daemon (multi-project)

A production `workflow.yaml` that runs entirely through the `teff` CLI —
no per-app `daemon.py`.  The workflow **is** the daemon: each tick walks
every project in `state.initial.project_ids`, lists open merge requests,
reviews each new one, posts the verdict back to GitLab and notifies
Telegram.  Already-reviewed MRs are tracked in `kv_store` under
`reviewed-<project>-<IID>` so they are never re-reviewed.

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
  `project_ids` list and clears the previous tick's conversation, so each
  tick starts clean even when durable state is checkpointed.
* `reviewer` — a ReAct agent with all tools scoped in.  It drives the whole
  review loop itself: for every project in `project_ids` it lists open MRs
  with `gitlab_list_open_mrs`, fetches diffs with `gitlab_get_mr_changes`,
  approves trivial MRs with `gitlab_approve`, comments with
  `gitlab_post_note` otherwise, records each verdict in `kv_store`, and
  sends a Telegram summary.  Because the agent drives the loop, the same
  workflow handles any number of projects.
* `tool_exec` — executes whatever tools the agent signalled and loops back
  until the agent stops calling tools.

## Requirements

* `teff` installed (this repo).
* A GitLab instance (self-hosted or gitlab.com) and a personal access
  token with `api` scope for the reviewed projects.
* A Telegram bot token and chat id.
* Ollama with `llama3.1:8b` (or point the workflow at any OpenAI-compatible
  provider via `react_agent` config).

## Configure

Env vars are interpolated into the workflow's tool configs (`${VAR}`).  Copy
`.env.example` and fill it in (or export the variables directly):

```bash
export GITLAB_URL=https://gitlab.example.com
export GITLAB_TOKEN=glpat-...
export TELEGRAM_BOT_TOKEN=123:abc
export TELEGRAM_CHAT_ID=-1001234567
```

Edit `workflow.yaml` and set the projects to track in
`state.initial.project_ids` (each is a path like `group/subgroup/repo` or
a numeric id):

```yaml
state:
  initial:
    project_ids:
      - "group/backend"
      - "group/frontend"
      - "1234"
```

## Run

```bash
# one tick (poll + review everything new), then exit
teff daemon -f workflow.yaml --once

# run forever, 60s between ticks
teff daemon -f workflow.yaml --interval 60
```

`kv_store` records already-reviewed MRs to `data/reviewed.json` and the
same tool instance lives across ticks, so dedup works without any
checkpoint.  Add `--checkpoint '{"type":"file","path":"data/daemon"}'` only
if you also want to persist counters and mid-review progress.

## Offline validation

```bash
teff validate workflow.yaml
```

## Tests

The core GitLab tools, the `tool_call` node and the multi-project workflow
loading are covered offline in `tests/` (mocked HTTP transports, no
network).
