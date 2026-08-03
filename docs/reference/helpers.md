# Top-level helpers

Small but useful functions exported from the top-level `draf` package. The
bigger surfaces are covered in their own pages: [Nodes](nodes.md),
[Tools](tools.md), [Providers](providers.md), [State guide](../guide/state.md).

## `set_defaults(provider=...)`

Set global defaults for subsequent `LLM`/`harness()` nodes (e.g. which
provider to use when a node doesn't name one):

```python
from draf import set_defaults

set_defaults(provider="ollama")
```

## `from_yaml(source)`

Parse a YAML string **or file path** into a compiled `Graph` (no tools —
tools come from `load_workflow`). Environment `${ENV}` interpolation is
applied:

```python
from draf import from_yaml

graph = from_yaml("""
steps:
  - id: start
    type: transform
    config: {action: uppercase, input_key: text, output_key: loud}
edges: []
""")
result = await graph.run({"text": "hi"})
```

For the full workflow loader (graph + tools + state + reducers) use
`draf.yaml.load_workflow` — see [YAML workflows](../guide/yaml-workflows.md).

## Typed state & reducers

State is a plain dict by default. For multi-writer keys (concurrent branches,
append-only conversation logs) use per-key merge strategies.

### `Reducer`

A merge strategy: `"override"` (default), `"append"` (list concatenation),
`"keep"` (first write wins), or a callable `(old, new) -> value`.

### `State`

A typed `dict` subclass that applies reducers extracted from a `TypedDict`
schema using `Annotated` metadata:

```python
from typing import Annotated, TypedDict
from draf import State


class MyState(TypedDict):
    messages: Annotated[list, "append"]
    status: str


state = State(MyState, {"status": "ok"})
state.merge({"messages": ["hello"]})
state.merge({"messages": ["world"]})
assert state["messages"] == ["hello", "world"]
```

### Reducer helpers

| Function | Purpose |
| -------- | ------- |
| `reducers_from_typeddict(cls)` | Extract reducers from a `TypedDict`'s `Annotated` metadata. |
| `reducers_from_yaml_schema(schema)` | Convert a YAML `state.schema` dict (`{key: {reducer: append}}`) into a reducer map. |
| `reducers_to_yaml_schema(reducers)` | Serialize a reducer map back to YAML (string reducers only). |
| `apply_reducers(state, new_values, reducers)` | Merge `new_values` into `state` using the given reducer map. |

In YAML:

```yaml
state:
  schema:
    messages: {reducer: append, type: list}
    status:   {reducer: keep}
```

## Schema utilities

| Function | Purpose |
| -------- | ------- |
| `json_schema_from_type(spec)` | Build a JSON Schema from a Python type spec — a raw schema dict, `dict[str, type]`, a `TypedDict`, or a dataclass. |
| `validate_json(value, schema)` | Validate a value against a JSON Schema; returns a list of human-readable error strings (empty = conforms). |

These back the `output_type`/`json_schema` feature of `LLM` — see
[Structured output](../guide/structured-output.md).

## `redact(value, keys=...)`

Redact credential-looking substrings from a value for safe logging. Used by
the tracer and cost reports so API keys never leak:

```python
from draf.errors import redact

redact("Bearer sk-1234-secret")  # -> "Bearer ***"
```

See also [Providers: cost & token reports](providers.md#cost--token-reports).

## Full export list

For the authoritative list of everything `draf` exports, see
[`draf/__init__.py`](../api/draf.md).

## The complete public surface

Auto-generated from docstrings, one page per module:

- [draf](../api/index.md) — top level: nodes, tools, graph, errors, pricing.
- [checkpoint](../api/draf.checkpoint.base.md) — `base`/`file`/`sqlite`/`pg`
  checkpointer classes.
- [trace](../api/draf.trace.md) — `RunTracer`, `TraceEvent`, `RunSummary`,
  `TokenUsage`, `tokens_cost`.
- [stream](../api/draf.stream.md) — `StreamEvent`.
- [eval](../api/draf.eval.md) — `run_eval`, `load_dataset`, `extract_output`.
- [prompt](../api/draf.prompt.md) — `render_template`.
- [yaml](../api/draf.yaml.md) / [yaml_schema](../api/draf.yaml_schema.md) —
  `load_workflow`, `workflow_to_yaml`, `validate_workflow(_file)`.
- [harness](../api/draf.harness.md) — `Harness`, provider concurrency.
- Every node, tool, RAG store, and plugin module.

Browse them all from the [API Overview](../api/index.md).