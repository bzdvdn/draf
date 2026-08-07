# YAML composition (`include:` + `loop` + `command`)

A pure-YAML workflow (no API key, no Ollama) that showcases the blocks for
assembling pipelines entirely from YAML:

- **`include:`** pulls in `shared_metrics.yaml` (prefixing its ids with
  `shared_`), which uses the new `split` and `contains` transform actions.
- **`command`** routes with `routes:` conditions — dynamic `goto` / `STOP`
  from state.
- **`loop`** re-runs a body chain until `attempts=3`.
- More `transform` actions: `to_int`, `replace`, `join`, `now`.

## Run

```bash
teff run --file examples/yaml_compose/workflow.yaml --pretty
# or
python examples/yaml_compose/run.py
```

The `include` path resolves relative to the workflow file, so the same file
works from any working directory.
