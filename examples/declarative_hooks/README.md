# declarative_hooks

Demonstrates the `hooks:` block: named hooks registered in a plugin are
referenced from the workflow YAML instead of being wired in Python.

- `plugins/trace_hooks.py` registers the `tick` hook with `@hooks.hook`.
- `workflow.yaml` loads the plugin (``plugins:``) and wires it via
  `hooks: {on_node_start: tick, on_node_end: tick}`.

```bash
teff run --file workflow.yaml
```

Observe a `[hook] ...` line for the start and end of each node. Add more
hooks (or a list, e.g. `on_node_end: [tick, finalize]`) — see
`docs/guide/yaml-workflows.md` and `docs/guide/best-practices.md`.