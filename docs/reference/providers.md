# Providers

`Harness` (and the `LLM` / `react()` / `harness()` nodes) speaks the OpenAI
chat-completions format by default and ships presets for several providers —
just pick a provider key and set the matching API key env var.

| `provider` | API key env var | Notes |
| ---------- | --------------- | ----- |
| `openai` | `OPENAI_API_KEY` | default |
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
from draf.harness import Harness

harness = Harness(
    model="claude-3-5-sonnet-latest",
    provider="anthropic",  # uses ANTHROPIC_API_KEY
    fallbacks=["gpt-4o"],  # fail over to another provider/model
)
reply = await harness.call([{"role": "user", "content": "hi"}])
print(reply.content, reply.cached, reply.latency_ms)
```

`Harness.from_config(cfg)` builds a harness from a node config dict (the same
keys `LLM` / `ReActAgent` accept), so Python and YAML stay in lockstep. In
YAML the `llm_chat` / `react_agent` nodes map `provider`, `base_url`,
`api_key_env`, `chat_path`, `fallbacks`, `cache`, `max_retries`, and the other
transport keys straight through.

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
from draf.harness import set_provider_concurrency, provider_concurrency

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
from draf import RunTracer

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
from draf import set_model_pricing, set_provider_pricing, model_pricing

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