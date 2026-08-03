# Durable execution (checkpoints)

`Graph.run()` accepts a `checkpointer` and a `checkpoint_id`. A checkpoint is
written **before** every node, so a crash or error resumes from the last safe
point instead of starting over.

```python
from draf import Graph
from draf.checkpoint import SQLiteCheckpointer
from draf.node import Transform

nodes = {"shout": Transform(action="uppercase", input_key="text", output_key="loud")}
graph = Graph(nodes, edges=[], entry_point="shout")
cp = SQLiteCheckpointer("checkpoints.db")

# first run crashes at some node
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")

# same id resumes from the saved checkpoint and completes
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")
```

## Backends

| Backend | Extra | Notes |
| ------- | ----- | ----- |
| `JSONFileCheckpointer` | core | file-based, `owner/` subdirectories |
| `SQLiteCheckpointer` | core | stdlib SQLite, composite `(owner, checkpoint_id)` key |
| `PGCheckpointer` | `draf[pg-checkpoint]` | needs PostgreSQL |

On resume the saved state wins over the passed-in state; a `State` instance
keeps its schema and reducers.

## Multi-tenant checkpoints (owner scoping)

Pass `owner=` to scope checkpoints to a user/session/tenant. The same
`checkpoint_id` under different owners never collides, so one store can serve
many users:

```python
await graph.run(state, checkpointer=cp, checkpoint_id="chat-1", owner="alice")
await graph.run(state, checkpointer=cp, checkpoint_id="chat-1", owner="bob")  # separate

chats = await cp.list("alice")  # ["chat-1", ...] — enumerate a user's runs
```

**Use `owner` for anything that should be isolated per end-user.** File
checkpoints land in an `owner/` subdirectory; SQLite/PG store a composite
`(owner, checkpoint_id)` primary key (existing single-owner databases migrate
automatically).

When `owner` is omitted, runs fall under the default owner `"default"`
(`draf.checkpoint.DEFAULT_OWNER`). The CLI exposes the same knob:
`--checkpoint-owner` on `draf run` and `draf inspect` (defaults to `default`).

## Human-in-the-loop (interrupts)

Pause a workflow for operator input with an `Interrupt` node. When execution
reaches it, `graph.run()` raises `GraphInterrupt`; resume with the same
`checkpoint_id` plus a `resume` dict:

```python
from draf.checkpoint import JSONFileCheckpointer
from draf.node.interrupt import GraphInterrupt
from draf.flow import Flow

flow = Flow("approval")
flow.step(LLM(model="llama3.1:8b", prompt="Составь план: {task}", output_key="draft"))
flow.interrupt(key="approved", prompt="Одобрить? (да / правки)")
flow.step(
    LLM(model="llama3.1:8b", prompt="{draft}\nВердикт: {approved}", output_key="final")
)

graph = flow.compile()
cp = JSONFileCheckpointer("checkpoints")

try:
    await graph.run(state=state, checkpointer=cp, checkpoint_id="run-1")
except GraphInterrupt as interrupt:
    print(interrupt.prompt)
    answer = input("> ")
    result = await graph.run(
        state=state,
        checkpointer=cp,
        checkpoint_id="run-1",
        resume={"approved": answer},
    )
```

The answer lands in `state["approved"]` and execution continues. Interrupts
require a `checkpointer`.

### Revision loop

Wire a conditional cycle back to the `Interrupt` with `Flow.loop()`:

```python
flow.step(LLM(model="llama3.1:8b", prompt="Составь план: {task}", output_key="draft"))
flow.interrupt(key="approved", prompt="Одобрить? (да / правки)")
flow.loop(
    key="approved",
    until="да",
    done=LLM(model="llama3.1:8b", prompt="{draft}", output_key="final"),
    body=LLM(
        model="llama3.1:8b",
        prompt="Переработай {draft} с учётом: {approved}",
        output_key="draft",
    ),
)
```

`loop()` wires `decider --key=until--> done` (stop) and
`decider --key!=until--> body -> decider` (repeat). `max_iterations` caps the
rounds. The decider can be any node that writes `key`, not just an `Interrupt`.