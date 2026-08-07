# Durable workflow: PostgreSQL checkpointer

Workflow with crash/resume persistence backed by PostgreSQL.

## Dependencies

Install the checkpoint extra:

```bash
uv add "teff[pg-checkpoint]"        # or: pip install "teff[pg-checkpoint]"
```

Start PostgreSQL. With Docker Compose (from `examples/checkpoint_stores/`):

```bash
docker compose up -d
```

Or manually:

```bash
docker run -d \
  -e POSTGRES_PASSWORD=postgres \
  -p 5433:5432 \
  pgvector/pgvector:pg16
```

The `failing` node in the workflow simulates a transient crash on the
first run (like a network blip or a timeout).

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/checkpoint_stores/run.py examples/checkpoint_stores/pg/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/checkpoint_stores/flow.py pg
```

## Notes

- The table `checkpoints` (configurable via `checkpoint.table`) is
  created lazily on first use.
- Connection string is `checkpoint.dsn` (default in the example:
  `postgresql://postgres:postgres@localhost:5433/postgres`); the compose
  file maps the container's 5432 to the host's 5433 so a locally running
  PostgreSQL is not disturbed.
- The example deletes the checkpoint first so the crash/resume dance is
  reproducible on every run.
