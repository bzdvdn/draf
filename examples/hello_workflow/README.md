# hello_workflow — the same workflow, three ways

A tiny **deterministic** workflow (no LLM, no network, no API key). It counts
the lines of a note and routes single-line vs multi-line input to the right
branch, then closes with `status: done`.

Three files each express the **exact same graph**, at three levels of
abstraction:

| File            | Abstraction        | Run it                                      |
| --------------- | ------------------ | ------------------------------------------- |
| `workflow.yaml` | Declarative (data) | `teff run --file workflow.yaml`             |
| `flow_dsl.py`   | Flow builder       | `uv run python flow_dsl.py`                 |
| `low_level.py`  | Low-level Graph    | `uv run python low_level.py`                |

Try them: single-line input yields `note: single-line note`, multi-line yields
`note: multi-line note`.

## The workflow

- `count` — a `Transform` node calls `count_lines` on `text` → `lines`.
- branch on `lines` — `lines=1` walks `single`, `lines!=1` walks `multi`.
- `status` (`Transform value: done`) is the converge point of both paths.

Read the shape in any one file and you can recognize it in the other two —
that's the whole point of choosing an abstraction level.

## Visualizing

`teff` graph the topology of the YAML as a Mermaid diagram:

```
teff graph examples/hello_workflow/workflow.yaml
```

## Tests

```
uv run pytest tests -q
```

They prove all three abstractions agree (single and multi-line), and that
`teff run --file` executes the YAML end-to-end.