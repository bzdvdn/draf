# Command (dynamic routing)

A `Command` is a node return value that does **two** things at once: it
merges a state update **and** names the node the graph should run next.
It is Draf's LangGraph-style dynamic edge — a node picks its successor at
runtime instead of being wired to fixed routes.

```python
from draf.node import Command

class AdminGate(Node):
    async def execute(self, ctx, state):
        if state.get("role") == "admin":
            return Command(update={"allowed": True}, goto="admin_tools")
        return Command(update={"allowed": False}, goto="denied")
```

## Why it exists

Normally a node returns a plain dict of state updates and routing is decided
by the graph's edges (`branch` / `loop` / string conditions on `step(...)`).
`Command` covers the cases edges can't express cleanly:

- the next node depends on the **result** of the node just executed;
- you want to **skip** intermediate nodes and jump to a specific node;
- you want to **terminate** the run from the middle of the graph.

## The three forms

| Returned value | Effect |
| -------------- | ------ |
| `Command(update={...}, goto="node_id")` | Merge `update`, then route to `node_id` (any node, even if no edge leads there). |
| `Command(update={...})` | Merge `update` only — routing follows the normal outgoing edges. |
| `Command(goto=Command.STOP)` | End the run immediately; the node's enclosing iteration stops. |
| `Command()` | A no-op: the node finishes without changing state or routing. |

### `goto`

`goto` accepts any node **id** in the graph — it does not need to be a
declared edge target. This is what makes it dynamic:

```python
async def route(self, ctx, state):
    if state.get("diag") == "fatal":
        return Command(goto="incident")
    return {"diag": state.get("diag")}   # normal edge routing
```

Routing to an **unknown** id raises `WorkflowError`. `Command.STOP` is a
sentinel that terminates the run (equivalent to a node with no outgoing
edge, but explicit).

### `update`

`update` is merged into state exactly like a plain dict return — per-key
[reducers](state.md) still apply. A `Command` with only `update` and no
`goto` behaves like returning the dict, so you can opt into a small,
readable node that both writes and routes.

### Sub-workflows

Inside a `Parallel`/`Map` branch only the `update` part is applied — the
`goto` is a top-level concern and is ignored there, so a branch can update
state but not hijack the parent's control flow.

## Plain functions can return it too

Any node accepted by `flow.step` — including a plain `(ctx, state) -> dict`
function — may return a `Command`:

```python
flow.step(lambda ctx, s: Command(update={"seen": True}, goto=Command.STOP))
```

Function nodes that return neither a dict nor a `Command` raise a `TypeError`.

## Example

See [examples/command_routing](../examples.md) for the CLI/offline walkthrough
(Flow + raw Graph), and the production-style FastAPI app
[examples/applications/fraud_gate](../examples.md) for a payment gate whose
router uses `Command` to auto-approve, send mid-risk payments to a human (a
durable `Interrupt`) or deny-and-stop — all driven by an LLM risk score.

```bash
python examples/command_routing/flow.py    # Flow API
python examples/command_routing/graph.py   # raw Graph API
```

## `STOP` doesn't mean error

Ending a run with `Command.STOP` is a normal, successful termination — the
final state is returned as usual. It differs from raising an error or the
`Interrupt` (which pauses and can be resumed).