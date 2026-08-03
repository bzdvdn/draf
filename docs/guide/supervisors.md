# Multi-agent supervisors (`route` + `agent_step`)

The most powerful (and most complex) pattern in draf is the **supervisor
loop**: one *decider* node picks which specialist agent handles the next
turn; that agent runs and control returns to the decider, which picks again —
until it says `finish` and the loop exits. This is what
[`repair-ai-chat`](../examples.md) and the `draf new` scaffolds build.

Two building blocks make it short:

- [`Flow.route()`](#flowroute) — the supervisor wiring (conditional edges +
  loops back to the decider).
- [`agent_step()`](#agent_step) — a framework helper that wraps one specialist
  agent (context builder → ReAct harness → append reply to conversation) as a
  `SubFlow`, ready to plug into `route()`.

## `Flow.route()`

Wires the **last added node** (the decider) into a supervisor loop. The
decider writes a routing key (e.g. `next_agent`); each keyword in `agents`
maps a value of that key to the chain run for it. After the chain finishes,
control returns to the decider. When the key equals `"finish"`, the loop
exits through the optional `finish` chain.

```python
from draf.flow import Flow
from draf.node import LLM

flow = Flow("support")
flow.step(
    LLM(
        model="llama3.1:8b",
        output_key="next_agent",
        system="Reply 'planner', 'estimator' or 'finish'.",
    )
)  # decider
flow.route(
    "next_agent",  # key the decider writes
    finish=final_llm,  # run when key == "finish"
    planner=planner_chain,  # chain for key == "planner"
    estimator=estimator_chain,  # chain for key == "estimator"
)
```

The wiring this produces:

```
decider --next_agent=planner--> planner-chain -> decider
decider --next_agent=estimator--> estimator-chain -> decider
decider --next_agent=finish--> finish-chain -> (continue)
```

Chains can be a single node, a list of nodes (run sequentially), or a
`SubFlow` (e.g. from `agent_step()`). `finish` is optional — when omitted the
flow terminates on `"finish"` and nothing can be chained after `route()`.

The decider is **any node that writes the routing key** — an `LLM` node, a
`react_agent`, an `Interrupt`, or a custom node. `route()` never cares what it
is, only that the previous `flow.step(...)` wrote `key`.

## `agent_step()`

One routed agent as a reusable `SubFlow`:

```
ContextBuilder ──► ReAct harness ──► AppendAssistant
```

- **ContextBuilder** composes a plain-text `input` from shared state sections
  (plus the latest user message) and resets the agent's scratch keys, so each
  run starts clean.
- The **harness** runs the model against that `input` with `use_tools`,
  writing its final answer to `output_key`.
- **AppendAssistant** copies that answer into the shared conversation
  (`messages_key`).

```python
from draf.flow import agent_step

planner = agent_step(
    "You are the planning agent. Produce a step list.",
    "plan",  # output_key in shared state
    model="llama3.1:8b",
    provider="ollama",
    sections={"draft": "Draft", "review": "Review"},  # context sections
    use_tools=["current_date"],  # allowlist, or "all" / None
    stream=True,
)
```

Arguments:

| Arg | Description |
| --- | ----------- |
| `system` | System prompt for the agent. |
| `output_key` | State key that receives the agent's final answer. |
| `model` / `provider` | LLM model and provider for the harness. |
| `sections` | Shared state key → label mapping rendered into the agent's context (default `{output_key: output_key.capitalize()}`). |
| `messages_key` | State key holding the shared conversation (default `"messages"`). |
| `use_tools` | `None`/`[]` (no tools, default), `"all"` (everything the pool offers), or an explicit allowlist of tool names. Prefer an allowlist. |
| `stream` | Emit tokens as stream events (live rendering, default `True`). |
| `**config` | Extra kwargs passed to the ReAct harness / `ToolExec`. |

`agent_chain` is a backwards-compatible alias. The agent's scratch conversation
lives in a private `_<output_key>_messages` state slot (reset by the context
builder); only the final reply reaches the shared `messages`.

## The full pattern (repair-ai style)

```python
from draf.flow import Flow, agent_step

flow = Flow("support")

flow.step(supervisor)  # decider writes "next_agent"
flow.route(
    "next_agent",
    planner=agent_step(
        PLANNER_PROMPT, "plan", model=model, provider=provider, sections=SECTIONS
    ),
    writer=agent_step(
        WRITER_PROMPT,
        "draft",
        model=model,
        provider=provider,
        sections=SECTIONS,
        use_tools=["current_date", "search_catalog"],
    ),
    reviewer=agent_step(
        REVIEWER_PROMPT, "review", model=model, provider=provider, sections=SECTIONS
    ),
)
graph = flow.compile()
```

Shared state keys (`plan`, `draft`, `review`) are rendered into every agent's
context via `sections`, so a later agent sees what earlier ones produced —
that is how writer + reviewer collaborate through the same conversation.

### Practical tips

- **Keep the loop bounded.** The decider's system prompt should explicitly
  list the route values and say `finish` ends the conversation, so the loop
  cannot spin forever.
- **Explicit tool allowlists.** Pass `use_tools=["name", ...]` per agent
  instead of `"all"` — it keeps `secret_tool` out of a specialist's reach.
- **One conversation, many agents.** Use a single `messages_key` (default
  `"messages"`) so every `agent_step` appends to the same thread.
- **Stream everything.** `agent_step(stream=True)` + `graph.stream()` renders
  tokens live while the supervisor routes between agents.

## Run it

- [`examples/simple_router/`](https://github.com/bzdvdn/draf/tree/main/examples/simple_router/)
  — a minimal two-agent supervisor, offline tests.
- [`examples/applications/repair-ai-chat/`](https://github.com/bzdvdn/draf/tree/main/examples/applications/repair-ai-chat/)
  — five routed agents with RAG + streaming.
- `draf new <name>` — scaffolds the same supervisor with `HOW TO EXTEND`
  comments (see [CLI](../cli.md)).

See also [`draf.flow.route`](../api/draf.flow.flow.md) and
[`draf.flow.agent_step`](../api/draf.flow.agent.md) in the API reference.