# {{PROJECT_NAME}}

A background-worker draf app scaffolded with
`draf new {{project_slug}} --template daemon`.  There is no HTTP surface:
producers drop a job (a JSON file) into `data/queue/`, and the worker polls
the directory, runs each job as one durable conversation turn through the
same supervisor `Flow`, writes the result to `data/results/` and removes the
job file.  Keep this skeleton generic and add your own agents, tools and
state — see **Add your own agent** below.

## Layout

```
{{project_slug}}/
├── daemon.py            # worker loop: poll queue, process turns, write results
├── src/                 # the production package
│   ├── config/          # env-driven settings (.env / DRAF_* vars)
│   ├── graphs/          # state, prompts, supervisor flow builder
│   ├── nodes/           # Supervisor + context builders
│   ├── queue/           # file-backed job queue (enqueue / pending / complete)
│   ├── tools/           # Tool subclasses handed to the agents
│   ├── service/         # Assistant: turn orchestration
│   └── storage/         # JSON-file checkpointer + session helpers
├── data/queue/          # pending jobs (created at runtime, git-ignored)
├── data/results/        # completed job results (created at runtime, git-ignored)
└── tests/               # wiring + queue tests (offline, no API keys)
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

## The job queue

Jobs are JSON files in `data/queue/`, each one a durable conversation turn:

```
data/queue/<job_id>.json   {"session_id": "...", "message": "..."}
data/results/<job_id>.json  final state, or {"error": "..."}
```

Producers call `enqueue(message, session_id=...)` from `src.queue`; the
worker polls with `daemon.py`, runs the turn, writes the result and removes
the pending job.  Re-use a `session_id` to continue an earlier conversation
(checkpointed in `data/checkpoints/`).

## Configuration

Settings live in `src/config/config.py` and are read from the environment
(`DRAF_*` vars) or a local `.env` file:

```
DRAF_PROVIDER=ollama
DRAF_MODEL=llama3.1:8b
DRAF_POLL_INTERVAL=2.0
```

## Add your own agent

Each piece carries a `HOW TO EXTEND` comment.  The usual loop:

1. Add a prompt to `src/graphs/prompts.py`.
2. Add an output slot to `AppState` in `src/graphs/state.py` (only if the
   agent produces shared state).
3. Build the agent chain in `src/graphs/build.py` with
   `agent_chain(system, output_key, use_tools)` and register it under a new
   keyword in the `route(..., **agents)` call.
4. Mention the new route value in the supervisor prompt, and (if the agent
   uses tools) add a `Tool` subclass in `src/tools/` and register it in
   `build_tools()`.

## Run

```
# 1. install the package (draf must be importable)
uv sync

# 2. offline tests — wiring + queue, no LLM required
uv run pytest tests/

# 3. drain the queue once and exit
uv run python daemon.py --once

# 4. poll forever (Ctrl-C to stop)
uv run python daemon.py
```

## Template placeholders

Rendered by `draf new`: `{{PROJECT_NAME}}`, `{{project_slug}}`,
`{{ProjectName}}`.
