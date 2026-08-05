# CLI Reference

`draf` runs YAML workflows, validates them, and reports on runs and evals.
It is installed with the package (`pip install draf`), or standalone via uv:

```bash
uv tool install draf         # global `draf` CLI
uvx draf -f workflow.yaml    # run on the fly without installing anything
```

```bash
draf -f workflow.yaml                      # run (the default command)
draf -f workflow.yaml --trace             # run + JSON trace to stderr
draf validate workflow.yaml               # validate without running
draf graph workflow.yaml                  # print the topology as YAML
draf graph workflow.yaml --mermaid        # render the graph as a Mermaid diagram
draf daemon -f workflow.yaml --once       # run one tick of a poll loop
draf daemon -f workflow.yaml --interval 60 # run forever, 60s between ticks
draf eval workflow.yaml --data dataset.jsonl --exact
draf inspect --checkpoint '{"type":"sqlite","path":"cp.db"}' --checkpoint-id run-1
draf prune --checkpoint '{"type":"file","path":"data/cp"}' --max-age 86400
draf new support-ai                       # scaffold a FastAPI app (default)
draf new support-cli --template cli       # scaffold a terminal-only app
draf new support-worker --template daemon # scaffold a background worker
draf new support-chat --template fastapi --with postgres,rag,celery  # + variants
draf obs-server --db traces.db --port 8001 # serve the trace dashboard + ingest
draf version
```

## obs-server

Serves the trace dashboard and an HTTP ingest endpoint. Workflows with an
`observability:` block (and no API of their own) push their traces here and
this process renders them:

```bash
draf obs-server --db ./data/traces.db --host 0.0.0.0 --port 8001
# open http://localhost:8001/obs/ui
```

- `--db` — SQLite file holding the traces (default `traces.db`).
- `--host` / `--port` — bind address / port (default `127.0.0.1:8001`).
- `--prefix` — URL prefix for the dashboard and ingest (default `/obs`).
- Ingest: `POST /obs/ingest` accepts a run in `Run.to_dict()` shape — the same
  body an `HttpExporter` (`type: webhook`) produces.
- Requires `draf[observability]` (fastapi + uvicorn).

## daemon

Re-runs a workflow on a poll interval (e.g. a GitLab reviewer), carrying state
between ticks via
`--checkpoint '{"type":"file","path":"data/cp"}'`.

## graph

Inspect a workflow's topology. Without flags it prints the normalized graph
as YAML; `--mermaid` renders a Mermaid flowchart instead (entry point
highlighted, edge conditions annotated, `__error__` edges styled):

```bash
draf graph workflow.yaml
draf graph workflow.yaml --mermaid
```

## inspect

Inspect a durable run by checkpoint:

```bash
draf inspect --checkpoint '{"type":"sqlite","path":"cp.db"}' \
  --checkpoint-id run-1 --checkpoint-owner default
```

`--checkpoint-owner` scopes the lookup to a tenant (defaults to `default`).

## prune

Delete stale checkpoints (TTL / keep-last GC) from any checkpointer backend:

```bash
draf prune --checkpoint '{"type":"file","path":"data/cp"}' --max-age 86400
draf prune --checkpoint '{"type":"sqlite","path":"cp.db"}' --keep-last 5
draf prune --checkpoint '{"type":"pg","dsn":"postgresql://..."}' \
  --checkpoint-owner alice --max-age 3600
```

`--max-age` removes checkpoints last written more than that many seconds ago;
`--keep-last` keeps only the N most recent per owner. `--checkpoint-owner`
restricts cleanup to one tenant; without it every owner is pruned. The command
prints how many checkpoints were removed and exits non-zero on errors.