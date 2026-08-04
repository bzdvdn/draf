# State

State is a flat, JSON-serializable dict. Nodes transform state — nothing else.

## Sharing keys across step configs

### Initial state

Declare a seed in `state.initial`:

```yaml
state:
  initial:
    title: "  hello world  "
```

### Parallel branches

`Flow.parallel()` runs independent branch chains concurrently, each branch
getting an isolated copy of the state. Per-key reducers merge updates back so
`append` branches accumulate instead of overwriting:

```python
from draf.flow import Flow
from draf.node import Transform

flow = (
    Flow("p")
    .parallel(
        [Transform(action="uppercase", input_key="title", output_key="title")],
        [Transform(action="uppercase", input_key="body", output_key="body")],
    )
    .converge(Transform(action="value", value="done", output_key="status"))
)

result = await flow.compile().run(state={"title": "hi", "body": "world"})
# -> title/body uppercased in parallel, then status="done"
```

Branches can be single nodes, lists of nodes (run sequentially inside the
branch), or embedded `Flow` subgraphs. The node also works directly:
`Parallel([[node1], [node2]])`.

### Dynamic fan-out (Map)

`Flow.map()` fans a state *list* into parallel branches at runtime — the
branch count comes from the data, not the declaration:

```python
flow = (
    Flow(
        "repair-plans",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
        default_model="llama3.1:8b",
    ).map(
        LLM(
            prompt="Составь план для ремонта {type} на сумму {summ} рублей.",
            output_key="plan",
        ),
        input_keys=["type", "summ"],  # lists zipped per index
        output_key="plans",  # list of per-item results
        max_concurrency=2,
    )
)
result = await flow.compile().run(
    state={
        "type": ["кухни", "санузел"],
        "summ": [150000, 80000],
    }
)
```

`chunk_size` batches items per branch, `max_concurrency` caps simultaneous
branches, `result_key` overrides the per-item key to collect.

## Reducers

Pass reducers to merges multi-writer keys deterministically. String reducers
(`append`, `replace`, …) round-trip to YAML via `reducers_to_yaml_schema()`.

## Prompt templates

LLM nodes read *multiple* state keys into one prompt with `{key}` templates
(also supported in `system`):

```python
node = LLM(
    model="llama3.1:8b",
    system="Ты инженер по ремонту.",
    prompt="Составь план для ремонта {type} на сумму {summ} рублей.",
    output_key="plan",
)
# state {"type": "кухни", "summ": 150000} -> user message:
# "Составь план для ремонта кухни на сумму 150000 рублей."
```

Values are stringified; a placeholder referencing a missing state key raises
`KeyError`. The underlying helper is `draf.prompt.render_template`.