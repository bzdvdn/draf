# Durable workflow: SQLite checkpointer

Workflow with crash/resume persistence backed by a SQLite database
(stdlib `sqlite3`, zero dependencies).

## Dependencies

- Nothing to install — Python's stdlib `sqlite3`.
- The `failing` node in the workflow simulates a transient crash on the
  first run (like a network blip or a timeout).

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/checkpoint_stores/run.py examples/checkpoint_stores/sqlite/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/checkpoint_stores/flow.py sqlite
```

## Notes

- Each checkpoint ID is one row; `save` is a single UPSERT transaction,
  so a crash leaves either the old or the new row, never a mix.
- The database lives in `./checkpoints.db` (configurable via
  `checkpoint.path`).
- The example deletes the checkpoint first so the crash/resume dance is
  reproducible on every run.
