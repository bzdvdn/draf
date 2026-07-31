# Durable workflow: JSON file checkpointer

Workflow with crash/resume persistence backed by JSON files — one file
per checkpoint ID.

## Dependencies

- Nothing to install — JSON and the file system.
- The `failing` node in the workflow simulates a transient crash on the
  first run (like a network blip or a timeout).

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/checkpoint_stores/run.py examples/checkpoint_stores/file/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/checkpoint_stores/flow.py file
```

## Notes

- Checkpoints are written atomically (temp file + rename), so a crash
  never leaves a corrupt checkpoint.
- Files live in `./checkpoints/` (configurable via `checkpoint.path`).
- The example deletes the checkpoint first so the crash/resume dance is
  reproducible on every run.
