# The minimal `route()` example

The smallest possible teff app built on
[`Flow.route()`](https://opencode.ai) — one supervisor that routes the
user's message to one of two simple agents, then exits when it says
`finish`.  It exists to show the *whole* supervisor pattern in one small
file set: no FastAPI, no tools, no domain layer.  Read
[`src/graphs/build.py`](src/graphs/build.py) first.

## The graph

```
supervisor ─ next_agent=coder ──► ContextBuilder → ReAct ──┐
    ▲                                                       │
    └────────────────────────── supervisor ◀───────────────┘
   (next_agent=talk)  ...  (next_agent=finish → exits)
```

The `Supervisor` LLM node writes `next_agent`; `route()` sends each value
to the matching agent chain (a `SubFlow`: context builder → ReAct harness
→ append the reply to the conversation) and loops back to the supervisor.
When it says `finish`, the loop exits.

## Why it can't hang

The supervisor loop is **bounded**: every call increments
`supervisor_rounds` in state, and once it reaches `max_rounds` the
supervisor forces `finish` without another LLM call.  A model that never
picks `finish` therefore cannot loop forever — the graph always
terminates.  The counter resets on each new user message (it lives in
`TRANSIENT_KEYS` in `src/storage/transient.py`).

It also stops early instead of burning that budget.  The `build.py` wiring
passes `done_keys={"code", "talk"}` with `done_mode="any"`, so once the
routed agent has written its answer the supervisor returns `finish`
deterministically — no second LLM call.  The `route_keys` guard also
ignores a pick that would re-run an agent whose slot is already filled.  The
decider's message now includes the work already produced plus the current
round, so the model sees what exists and routes (or finishes) accordingly.

## Layout

```
simple_router/
├── cli.py               # run one turn on the terminal
├── src/
│   ├── config/          # env-driven settings (.env / TEFF_* vars)
│   ├── graphs/          # state, prompts, supervisor flow builder
│   ├── nodes/           # context builders (Supervisor comes from teff.node)
│   ├── service/         # Assistant: one durable turn
│   └── storage/         # JSON-file checkpointer + session helpers
└── tests/               # offline tests (no LLM, no network)
```

## Run

Run from the repository root (the repo's ``uv`` project resolves ``teff``;
the tests and scripts add their own root to ``sys.path``).  Requires Ollama
running locally for the live turn:

```
# 1. offline tests — wiring + route loop + bounded termination
uv run pytest examples/simple_router/tests/

# 2. run one turn against a local Ollama (streams tokens)
uv run python examples/simple_router/cli.py run "write a python one-liner to list files"
```

## Add your own agent

Each piece carries a `HOW TO EXTEND` comment.  The usual loop:

1. Add a prompt to `src/graphs/prompts.py`.
2. Add an output slot to `RouterState` in `src/graphs/state.py` (only if
   the agent produces shared state).
3. Build the agent chain in `src/graphs/build.py` with
   `agent_step(system, output_key)` and register it under a new keyword
   in the `route(..., **agents)` call.
4. Mention the new route value in the supervisor prompt.
