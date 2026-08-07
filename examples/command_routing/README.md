# Command routing

[`Command`](../../teff/node/command.py) is a node return value that combines
a **state update** with an explicit **next-node route** — LangGraph-style
dynamic routing built into Teff's `Flow` and `Graph`.

Normally a node returns a plain dict of updates and the graph follows its
outgoing edges. Returning a `Command` lets the node *also* pick the next node
at runtime:

- `Command(update={...}, goto="node_id")` — merge `update` **and** jump to
  any node in the graph, bypassing every edge in between.
- `Command(update={...})` (no `goto`) — merge `update`, keep normal edge
  (condition) routing.
- `Command(goto=Command.STOP)` — merge nothing, end the run immediately.
- `Command()` — no-op result that lets a node finish without touching state.

This example is a small content-moderation gate with **no LLM, no API key,
no Ollama** — it runs offline and each path demonstrates one of the
behaviours above:

| input            | behaviour shown                          |
| ---------------- | ---------------------------------------- |
| `"this is bad"`  | `goto=Command.STOP` — run ends, `blocked=True` |
| `"trusted user"` | `goto="deliver"` jumps past `review` (no such edge exists) |
| `"...?"`         | `Command(update=...)` — keep normal edge → `review` |
| `"hello"`        | plain dict — normal edge routing |

## Run

```bash
python examples/command_routing/flow.py    # Flow API
python examples/command_routing/graph.py   # raw Graph API
```

Both build the same workflow — `flow.py` with the `Flow` builder,
`graph.py` with explicit `Graph` + `Edge` — and print the resulting route
for each input.