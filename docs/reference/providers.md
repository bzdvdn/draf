# Providers

A provider is a named model endpoint: how to speak to it (`type` — see
`Provider` below) and where it lives (`base_url` / `chat_path` / auth keys).
`Harness`, `LLM`, `react()` and `supervisor()` nodes route model calls through
a provider. Built-in **presets** (subclasses) carry the defaults for the
common providers; you declare exactly which ones you use.

| preset | API key env var | Notes |
| ------ | --------------- | ----- |
| `openai` | `OPENAI_API_KEY` | |
| `anthropic` | `ANTHROPIC_API_KEY` | responses normalised to OpenAI shape |
| `deepseek` | `DEEPSEEK_API_KEY` | |
| `mistral` | `MISTRAL_API_KEY` | |
| `together` | `TOGETHER_API_KEY` | |
| `groq` | `GROQ_API_KEY` | |
| `openrouter` | `OPENROUTER_API_KEY` | |
| `gemini` | `GEMINI_API_KEY` | Google's OpenAI-compatible endpoint |
| `openai_compatible` | `OPENAI_API_KEY` | any custom endpoint (vLLM, LM Studio, Azure) |
| `ollama` | — (local) | |

## Using a provider

```python
from teff.harness import Harness

harness = Harness(
    model="claude-3-5-sonnet-latest",
    provider="anthropic",  # uses ANTHROPIC_API_KEY
    fallbacks=["gpt-4o"],  # fail over to another provider/model
)
reply = await harness.call([{"role": "user", "content": "hi"}])
print(reply.content, reply.cached, reply.latency_ms)
```

A bare, standalone `Harness` (`providers=None`) falls back to the built-in
preset matching `provider`. Anywhere a graph or workflow supplies a
`providers=` map / registry, the rule is **strict**: a provider is only usable
after it has been explicitly declared there (see below).

`Harness.from_config(cfg)` builds a harness from a node config dict (the same
keys `LLM` / `ReActAgent` accept), so Python and YAML stay in lockstep. In
YAML the `llm_chat` / `react_agent` nodes map `provider`, `base_url`,
`api_key_env`, `chat_path`, `fallbacks`, `cache`, `max_retries`, and the other
transport keys straight through.

## Resolving provider and model

There is no global default provider *or* model. Per node, the resolution is:

1. the **provider**: the node's explicit `provider=`, else the graph-level
   `default_provider=` (`Graph(...)`, `Flow("...", default_provider=...)`, or
   a workflow's top-level `default_provider:`);
2. the **model**: the node's explicit `model=`, else the graph-level
   `default_model=`.

If neither is set, the node raises `ConfigError` — there is no silent fallback
(no implicit `"gpt-4"`). The resolved provider must be *declared* in the
`providers=` map / `providers:` block (see below).

```python
from teff.flow import Flow
from teff.provider import ProviderRegistry

flow = Flow(
    "repair",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.llm()  # inherits provider="ollama" and default_model="llama3.1:8b"
```

A node can still override the graph default with its own `provider=` / `model=`.

## The `Provider` value object

Under the hood each provider is a `Provider` — a lightweight value object that
picks the wire protocol and the endpoint:

```python
from teff import Provider

Provider(type="anthropic_compatible", base_url="http://proxy", chat_path="/v1/messages")
```

`type` is the protocol discriminator — `openai_compatible`,
`anthropic_compatible`, or `ollama` — and decides the request body, streaming
chunk parsing, and response extraction. The name is just a key; it never
carries protocol meaning.

## Declaring providers (strict)

A graph's `providers=` map / `providers:` block is the **single source of
truth**. Providers used on nodes, by `default_provider`, or by
`default_model` must be explicitly declared — there is no implicit built-in
fallback. There is **no string shorthand**: pass real `Provider` instances (or
a `ProviderRegistry`), never bare preset-name strings.

```python
from teff import Provider
from teff.graph import Graph
from teff.node import LLM

providers = {
    "vllm": Provider(base_url="http://vllm:8000/v1"),  # openai_compatible
    "claude": Provider(type="anthropic_compatible", base_url="http://proxy"),
}

graph = Graph({"llm": LLM(model="m", provider="claude", api_key_env="X")}, [], "llm")
result = await graph.run({}, providers=providers)
```

`providers=` at run time wins over `graph.providers`. Any provider referenced
but not declared — a node `provider=`, the `default_provider=`, or a
`providers:`/`providers=` name — raises `ConfigError`, so typos surface early
instead of silently routing to the wrong wire protocol.

### Register providers with `ProviderRegistry`

A `ProviderRegistry` is a dict-like `{name: Provider}` map you build once and
reuse. It starts empty, and only registered names are usable.

```python
from teff import Graph, Provider, ProviderRegistry
from teff.flow import Flow
from teff.harness import Harness
from teff.node import LLM

reg = ProviderRegistry()  # register instances explicitly…
reg.register(Provider(name="vllm", base_url="http://vllm:8000/v1"))
reg.register(
    Provider(name="claude", type="anthropic_compatible", base_url="http://proxy")
)
# …and reference entries by name anywhere:
graph = Graph({"llm": LLM(model="m", provider="claude")}, [], "llm", providers=reg)
result = await graph.run({})  # registry also accepted per-run: run({}, providers=reg)

flow = Flow("f", providers=reg)  # threaded into the compiled graph
graph2 = flow.step(LLM(model="m", provider="claude")).compile()

harness = Harness(model="m", provider="claude", providers=reg)
```

Register the built-ins you use explicitly with
`ProviderRegistry.from_presets("openai", "ollama")` — it instantiates each
named preset, so `graph.providers` truthfully reflects what is configured.
`register()` returns the registry so registrations can be chained. Duplicate
names (and names whose `Provider.name` mismatches the registry key) raise
`ConfigError`. A plain `dict` is normalised automatically, and
`to_provider_registry(x)` converts a `dict`, `None`, or existing registry to a
`ProviderRegistry` explicitly.

## Secrets via `${ENV}`

Never hardcode keys. In workflows use the environment interpolation:

```yaml
steps:
  - type: llm_chat
    config:
      provider: openai
      model: gpt-4o
      api_key_env: ${OPENAI_API_KEY}
```

## Concurrency caps

`set_provider_concurrency(provider, limit)` caps concurrent model calls
globally per provider — across every `Harness` instance, so parallel branches
throttle together instead of blowing past provider rate limits:

```python
from teff.harness import set_provider_concurrency, provider_concurrency

set_provider_concurrency("openai", 8)  # global cap
provider_concurrency("openai")  # -> 8 (the active cap, or None)
set_provider_concurrency("openai", 0)  # remove the cap
```

## Response caching

`cache=True` dedupes model calls: the request body is hashed and a cached
reply is returned for identical re-calls (checkpoint resumes and eval re-runs
never pay twice). Cached replies report `ModelReply.cached == True`. Pass a
custom mutable mapping instead of a plain dict; streaming responses are not
cached.

```python
harness = Harness(model="gpt-4o", cache=True)
first = await harness.call(messages)  # network round-trip
second = await harness.call(messages)  # served from cache
assert first.cached is False and second.cached is True
```

## Cost & token reports

`RunSummary` folds the trace into cost and token figures. Costs use an
internal model-price table (exact name match, then prefix match; unknown and
local models cost $0). Secrets are redacted from every reported value.

```python
from teff import RunTracer

tracer = RunTracer()
await graph.run(state, tracer=tracer)

summary = tracer.summary()
print(summary.cost_usd)  # estimated spend in USD
print(summary.tokens)  # prompt/completion totals
summary.to_dict()  # JSON-serialisable (redacted)
summary.to_json()
```

### Custom pricing per provider / model

Register custom USD-per-1M-token pricing at runtime; it takes precedence over
the built-in table:

```python
from teff import set_model_pricing, set_provider_pricing, model_pricing

set_model_pricing("openrouter", "openai/gpt-4o", 3.0, 12.0)
set_provider_pricing("kilo", 0.1, 0.4)  # whole-provider default
print(model_pricing("openai/gpt-4o", "openrouter"))  # (3.0, 12.0)
```

Or load a whole file at once — `load_pricing("pricing.yaml")` (or a dict):

```yaml
providers:
  openrouter:
    default: {input: 0.1, output: 0.4}
    models:
      "openai/gpt-4o": {input: 3.0, output: 12.0}
```

Resolution order: exact `(provider, model)` → provider-prefixed custom entry
→ provider-wide default → built-in table → $0 for unknown/local models.
`clear_pricing()` resets everything to the built-in table.

For a one-off estimate from token counts, use `tokens_cost(model,
prompt_tokens, completion_tokens, provider="")` — it applies the same
pricing table and returns the USD figure.