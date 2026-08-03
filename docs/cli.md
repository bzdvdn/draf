# CLI Reference

`draf` runs YAML workflows, validates them, and reports on runs and evals.

```bash
draf -f workflow.yaml                      # run (the default command)
draf -f workflow.yaml --trace             # run + JSON trace to stderr
draf validate workflow.yaml               # validate without running
draf daemon -f workflow.yaml --once       # run one tick of a poll loop
draf daemon -f workflow.yaml --interval 60 # run forever, 60s between ticks
draf eval workflow.yaml --data dataset.jsonl --exact
draf inspect --checkpoint '{"type":"sqlite","path":"cp.db"}' --checkpoint-id run-1
draf new support-ai                       # scaffold a FastAPI app (default)
draf new support-cli --template cli       # scaffold a terminal-only app
draf new support-worker --template daemon # scaffold a background worker
draf new support-chat --template fastapi --with postgres,rag,celery  # + variants
draf version
```

## daemon

Re-runs a workflow on a poll interval (e.g. a GitLab reviewer), carrying state
between ticks via
`--checkpoint '{"type":"file","path":"data/cp"}'`.

## inspect

Inspect a durable run by checkpoint:

```bash
draf inspect --checkpoint '{"type":"sqlite","path":"cp.db"}' \
  --checkpoint-id run-1 --checkpoint-owner default
```

`--checkpoint-owner` scopes the lookup to a tenant (defaults to `default`).