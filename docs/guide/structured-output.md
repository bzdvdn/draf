# Structured output

Guarantee the LLM returns a schema-conforming JSON object instead of free
text. Pass a JSON Schema (`json_schema`) or a Python type spec (`output_type`
— `TypedDict`, dataclass, or `dict[str, type]`):

```python
from typing import TypedDict
from draf.flow import Flow
from draf.node import LLM
from draf.provider import ProviderRegistry


class Weather(TypedDict):
    city: str
    temp: float


flow = Flow(
    "weather",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.step(LLM(output_key="weather", output_type=Weather))
graph = flow.compile()

result = await graph.run({"city": "Москва"})
result["weather"]  # {"city": "...", "temp": 12.5} — a parsed dict, validated
```

The response is parsed as JSON, validated against the schema, and re-asked
with the validation error fed back (up to `max_retries`, default 2). If all
attempts fail, a `StructuredOutputError` is raised — route it with an
`__error__` edge. Schema errors are recorded as `structured` events in the
tracer and the stream.

Without a schema, `parse=True` still parses the response into a dict (no
validation):

```python
flow.step(LLM(output_key="data", parse=True))
```

## In YAML

The same field map works declaratively (with the provider declared at the
top):

```yaml
name: weather
default_provider: ollama
providers:
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    chat_path: /api/chat
steps:
  - id: weather
    type: llm_chat
    config:
      model: llama3.1:8b
      output_key: weather
      json_schema: {type: object, properties: {city: {type: string}, temp: {type: number}}, required: [city, temp]}
```

## Re-asking on failure

A model that emits invalid JSON is prompted again with the validation error.
Tune it with `max_retries` on the node; on final failure the run raises
`StructuredOutputError`, which you can catch or route:

```python
from draf import StructuredOutputError

try:
    result = await graph.run(state)
except StructuredOutputError as exc:
    print(exc.schema)
```