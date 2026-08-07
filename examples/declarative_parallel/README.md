# declarative_parallel

Demonstrates the `parallel` node expressed **directly in YAML** (before this
feature, parallel branches required the Python `Flow.parallel(...)` builder).

`workflow.yaml` fans one input out to three concurrent `transform`
branches; each writes its own state key, and the lines merge back before a
final `report` node.

```bash
teff run --file workflow.yaml
```

Output (keys `upper`, `lines`, `trimmed`, `status`):

```json
{
  "text": "  hello teff  ",
  "upper": "  HELLO Teff  ",
  "lines": "1",
  "trimmed": "hello teff",
  "status": "done"
}
```

A branch can also be a **list** of sequential steps — nest a block sequence
under `branches` (see the parallel section in `docs/guide/yaml-workflows.md`).