# declarative_checkpoint

Demonstrates the `checkpoint:` block: durable runs whose state is persisted
before every node.

The workflow upcases a draft, pauses on a `gate` interrupt, and ships. The
`checkpoint:` block points at a SQLite file (path resolved relative to this
folder).

Run it with durable runs and resume:

```bash
# half a run: upcase → gate, then pause and answer "yes"
teff run --file workflow.yaml --checkpoint-id demo --interactive

# resume from saved state (the gate answer), or continue an existing run
teff run --file workflow.yaml --checkpoint-id demo
```

State written to `data/checkpoints.db`. The progress survives process
restarts — a crash mid-run resumes rather than starting over. See
`docs/guide/yaml-workflows.md` and `docs/guide/durable.md`.