# Durable workflow examples

The same crash/resume workflow running on every checkpoint backend teff
supports.  The graph is three nodes — a transform, a node that fails
once (simulating a transient crash), and another transform.  The first
run crashes; a re-run with the same `checkpoint_id` resumes from the
saved checkpoint and completes.

| Backend   | Deps needed                | Server required | Persistence    | Best for                          |
| --------- | -------------------------- | --------------- | -------------- | --------------------------------- |
| `file`    | none (core)                | no              | yes (files)    | simple, local, per-thread files   |
| `sqlite`  | none (stdlib `sqlite3`)    | no              | yes (file)     | single-file durable state         |
| `pg`      | `teff[pg-checkpoint]`      | yes (PostgreSQL)| yes            | shared/durable at scale           |

Each subdirectory has its own README with the exact install steps:

- [file](file/README.md)
- [sqlite](sqlite/README.md)
- [pg](pg/README.md)

## How it works

`Graph.run()` accepts a `checkpointer` and a `checkpoint_id`.  A
checkpoint is written **before** every node execution, so:

- a **crash** mid-node resumes by re-running that node;
- a node that **raises** and routes via an `__error__` edge resumes from
  the fallback, not the failed node;
- a completed run stores a terminal checkpoint (`next_node_id: null`),
  so re-running the same ID returns the final state.

On resume the saved state wins over the passed-in state, and a `State`
instance keeps its schema and reducers.

## Shared prerequisites

The `pg` backend needs the extra and a running server; `file` and
`sqlite` need nothing.

```bash
uv add "teff[pg-checkpoint]"        # or: pip install "teff[pg-checkpoint]"
```

Start PostgreSQL for `pg` (from `examples/checkpoint_stores/`):

```bash
docker compose up -d
```

## Run any of them

Two entry points — same workflow, same crash/resume dance.

`run.py` loads the workflow from `workflow.yaml`, including its
`checkpoint:` block (CLI emulation):

```bash
uv run python examples/checkpoint_stores/run.py examples/checkpoint_stores/file/workflow.yaml
uv run python examples/checkpoint_stores/run.py examples/checkpoint_stores/sqlite/workflow.yaml
uv run python examples/checkpoint_stores/run.py examples/checkpoint_stores/pg/workflow.yaml
```

`flow.py` builds the same graph with the Python Flow API (pass the
backend as an argument):

```bash
uv run python examples/checkpoint_stores/flow.py file
uv run python examples/checkpoint_stores/flow.py sqlite
uv run python examples/checkpoint_stores/flow.py pg
```

Expected output (every backend):

```
Run 1: crashed (simulated transient failure), checkpoint saved
Run 2: success -> {'text': 'durable', 'shout': 'DURABLE', 'recovered': True, 'status': 'finished'}
```
