# The minimal `route()` example — pure YAML

The same router as [`simple_router`](../simple_router/README.md), but written
entirely as data: **one `workflow.yaml`, zero graph-building Python.**  It is
the smallest possible declarative supervisor — no FastAPI, no tools, no domain
layer.

## The graph

```
supervisor ─ next_agent=coder ──► coder (subflow: agent_step) ──┐
    ▲                                                           │
    └────────────────────────── supervisor ◀───────────────────┘
   (next_agent=talk)  ...  (next_agent=finish → exits)
```

`supervisor` is the `supervisor` node type; `coder`/`talk` are `subflow` steps
whose `config.build` reuses the `agent_step` recipe (context builder → ReAct
harness → append the reply to `messages`).  The edges are plain data:

```yaml
edges:
  - from: supervisor
    to: coder
    condition: next_agent=coder
  - from: supervisor
    to: talk
    condition: next_agent=talk
  - from: coder
    to: supervisor
  - from: talk
    to: supervisor
```

## Why it can't hang

Everything the code version gets from `Flow.supervisor()` is config here:

- `max_rounds: 6` — the loop budget; once `supervisor_rounds` reaches it the
  supervisor returns `finish` without another model call.
- `done_keys: [code, talk]`, `done_mode: any` — as soon as one agent has
  written its answer the supervisor finishes deterministically.
- `route_keys: {coder: code, talk: talk}` — a pick that would re-run a filled
  agent slot is ignored.
- `fallback_agent: talk` — a premature `finish` still produces a real answer.

`state.schema` gives `messages` an `append` reducer so the conversation
accumulates across the routed agents.

## Run

From the repository root (requires Ollama running locally for the live turn):

```
# 1. offline tests — validation + wiring + bounded termination
uv run pytest examples/simple_router_yaml/tests/

# 2. run one turn against a local Ollama (streams tokens)
uv run draf -f examples/simple_router_yaml/workflow.yaml

# 3. inspect the topology
uv run draf graph examples/simple_router_yaml/workflow.yaml --mermaid
```

## Compare with the code version

`examples/simple_router` builds the identical graph with
`Flow.supervisor(...)` + `Flow.route(...)`; `workflow.yaml` here is its
`workflow_to_yaml()` output by hand.  Same prompts, same guards, same
termination guarantees — declared instead of constructed.
