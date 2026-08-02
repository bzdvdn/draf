# {{PROJECT_NAME}}

A terminal-first draf app scaffolded with `draf new {{project_slug}} --template cli`.
Same supervisor `Flow` as the `fastapi` template, but the production
interface is the command line instead of an HTTP server — there is no
network service at all.  Sessions are still durable via the JSON-file
checkpointer.  Keep this skeleton generic and add your own agents, tools
and state — see **Add your own agent** below.

## Layout

```
{{project_slug}}/
├── cli.py               # typer app: run + chat subcommands
├── src/                 # the production package
│   ├── config/          # env-driven settings (.env / DRAF_* vars)
│   ├── core/            # composition root: src/core/container.py build_container
│   ├── graphs/          # state, prompts, supervisor flow builder
│   ├── nodes/           # Supervisor + context builders
│   ├── tools/           # Tool subclasses handed to the agents
│   ├── service/         # Assistant: turn orchestration (CLI)
│   └── storage/         # JSON-file checkpointer + session helpers
├── data/checkpoints/    # durable session state (created at runtime, git-ignored)
└── tests/               # wiring + CLI tests (offline, no API keys)
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
DRAF_CHECKPOINT_DIR=            # empty = data/checkpoints
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

# 2. offline tests — wiring + CLI, no LLM required
uv run pytest tests/

# 3. run one turn against a local Ollama (streams tokens)
uv run python cli.py run "Help me draft a note"

# 4. interactive chat session
uv run python cli.py chat
```

## Template placeholders

Rendered by `draf new`: `{{PROJECT_NAME}}`, `{{project_slug}}`,
`{{ProjectName}}`.
