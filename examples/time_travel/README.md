# Time travel: rewind a finished run to any checkpoint and replay

A story-writing graph runs to completion, saving a checkpoint *before each
node* via a checkpointer. Time travel lets you:

1. Inspect the checkpoint history (which node runs next at each step).
2. Jump back to any earlier checkpoint.
3. Edit the state at that moment.
4. Replay from there — the past (chapters 1-2) stays identical, the
   future (chapters 3-4) is rewritten.

```
setup -> conflict -> twist -> ending
```

This mirrors LangGraph's "time travel" (rewind a thread to a checkpoint,
edit state, and resume), built on DRAFTFLOW's `SQLiteHistoryCheckpointer`.

`SQLiteHistoryCheckpointer` extends `SQLiteCheckpointer` with a
`checkpoint_history` table: every save (one per node, keyed by iteration)
is appended, so an earlier checkpoint survives the overwrite of the
"current" one. The story chapters are written by built-in `Transform`
nodes with `action: append` — the "accumulate formatted text" pattern.

## Files

| File           | What it shows                                                                    |
| -------------- | -------------------------------------------------------------------------------- |
| `graph.py`     | The four chapter steps wired by hand with the low-level `Graph` API              |
| `flow.py`      | The same linear chain with the `Flow` builder — four `flow.step()` calls, no explicit edges |
| `workflow.yaml`| The same story as pure YAML — four `transform` appends, runnable with `draf run` |

All three are fully self-contained, share the exact same
checkpointing/time-travel logic (which comes entirely from the framework),
and need no LLM.

## Run it

Fully offline — nodes are plain `Transform` appends, no LLM needed, no API
key:

```bash
uv run python examples/time_travel/graph.py
uv run python examples/time_travel/flow.py
```

Expected output (both Python files):

```
=== Rewound to iter 2, changed twist -> it was all a dream ===
past preserved (chapters 1-2): True
future rewritten (chapters 3-4): True
```

### The YAML variant

`workflow.yaml` describes just the story graph. Time travel is about the
*store*, so run it with a history checkpointer to write the per-step
timeline:

```bash
draf run -f examples/time_travel/workflow.yaml \
  --checkpoint '{"type":"sqlite_history","path":"cp.db"}' \
  --checkpoint-id story
```

The `cp.db` history table now holds every snapshot, so a later script can
rewind, edit, and replay — the same interaction shown in `graph.py`.

## Production

`PGHistoryCheckpointer` provides the same time travel on PostgreSQL
(`draf[pg-checkpoint]`), and the CLI can pick either via `--checkpoint`:

```bash
# SQLite history store
draf run -f workflow.yaml --checkpoint '{"type":"sqlite_history","path":"cp.db"}'
# PostgreSQL history store
draf run -f workflow.yaml --checkpoint '{"type":"pg_history","dsn":"postgresql://..."}'
```

The time-travel *interaction* (rewind, edit, replay) is Python-only —
it needs a real edit-and-replay loop — but the storage backend is fully
declarative.

