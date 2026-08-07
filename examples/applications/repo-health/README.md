# Repo-health daemon (workflow.yaml)

A production `workflow.yaml` that runs entirely through the `teff` CLI —
no per-app Python.  Each tick the **agent is the driver**: it takes a
distributed lock, reads a priority table from a CSV, inspects a git
repository, cross-references changed files against the CSV, de-duplicates
alerts in redis, waits for a maintenance-window signal and sends a single
Telegram summary.

This example is built on the coordination tools in the core: `git`,
`lock`, `wait_for`, `csv_query`, `redis`.

> The same logic is also available as a programmatic `flow.py` (built with
> `teff.flow.Flow`).  Choose whichever fits your setup — both run the exact
> same tools and steps.

## How it works

```
teff daemon -f workflow.yaml --interval 300
```

```
reset (context_builder) ──► agent (react_agent) ──► tool_exec ──► (loop)
                                  ▲                        │
                                  └──────── _tool_call_name!= ┘
```

* `reset` — rebuilds the agent's `input` (with the priority CSV path) and
  clears the previous tick's conversation, so each tick starts clean even
  with `--checkpoint`.
* `agent` — a ReAct agent with all tools scoped in.  It drives the whole
  tick itself, in order:
  1. `lock acquire daemon:tick` — skip the tick if another instance holds it.
  2. `csv_query read` the priority table.
  3. `git status` + `git log` — what changed recently.
  4. `csv_query filter` — owner/priority for each changed file.
  5. `redis exists` / `redis set` — alert each file only once a day.
  6. `wait_for redis_key deploy:ready` — wait for the maintenance window
     (a timeout is reported, not fatal).
  7. `send_telegram` — one summary.
  8. `lock release daemon:tick`.
* `tool_exec` — executes whatever tools the agent signalled and loops back
  until the agent stops calling tools.

Because the agent drives the loop, adding a step to a tick is just editing
the prompt — no code.

## Requirements

* `teff` installed (this repo), with `teff[tools]` for `redis`.
* A Redis-compatible server (Redis, KeyDB, Valkey) for `lock`/`redis`/
  `wait_for` — point `REDIS_URL` at it.
* A Telegram bot token and chat id.
* A git repository to watch (the `git` tool `path`).
* Ollama with `llama3.1:8b` (or point the workflow at any OpenAI-compatible
  provider via `react_agent` config).

## Configure

Env vars are interpolated into tool configs (`${VAR}`).  Copy
`.env.example` and fill it in:

```bash
export REDIS_URL=redis://localhost:6379/0
export TELEGRAM_BOT_TOKEN=123:abc
export TELEGRAM_CHAT_ID=-1001234567
```

Point `git.path` at the repository to watch and `csv_query.path` at the
priority table in `workflow.yaml`:

```yaml
tools:
  - type: git
    config:
      path: "/srv/acme/backend"      # repository root to inspect
  - type: csv_query
    config:
      path: "data/priority.csv"      # file,owner,priority
```

`data/priority.csv` maps file paths to owners and priorities
(`high`/`medium`/`low`) so the agent knows which changes matter.

## Run

Two equivalent ways to run the same tick.

**1. Daemon from YAML (workflow as data):**

```bash
# one tick, then exit
teff daemon -f workflow.yaml --once

# run forever, every 5 minutes
teff daemon -f workflow.yaml --interval 300
```

`redis` keeps the `alerted:<file>` keys, so de-dup survives restarts
without any checkpoint.  Add
`--checkpoint '{"type":"file","path":"data/daemon"}'` only if you also want
to persist counters and mid-tick progress.

**2. Programmatic flow (Python):**

```bash
# one tick, then exit
python flow.py
```

`flow.py` builds the identical graph with `teff.flow.Flow` (a
`context_builder` step plus a ReAct loop) and constructs the same tool
instances from the same env vars (`REDIS_URL`, `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`).  It reads the same `data/priority.csv` and `git`
`path`, so switching between YAML and Python is just a matter of how you
launch it.  To run it as a repeating daemon from Python, wrap
`flow.main()` in your own `while True` loop with the interval you want.

## Offline validation

```bash
teff validate workflow.yaml
```

## Tests

The underlying tools (`git`, `lock`, `wait_for`, `csv_query`, `redis`) and
the loading/validation of this workflow are covered offline in `tests/`
(mocked subprocess / redis / httpx, no network).
