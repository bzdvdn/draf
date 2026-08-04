# YAML workflows

The canonical graph is YAML/JSON — code is optional. Workflows are loaded with
`load_workflow`, validated with `draf validate`, and serialized back out with
`workflow_to_yaml` / `Flow.to_yaml()`.

## Structure

```yaml
name: my-workflow            # optional label
description: ...             # optional

state:
  initial: {title: hello}    # seed values

plugins: [nodes]             # optional: extra node/tool modules to import
plugins_folder: plugins      # optional auto-loaded folder (default: plugins)

tools:                       # tool instances made available to agents
  - type: web_search

steps:                       # nodes
  - id: search
    type: web_search
    config: {query_key: q, output_key: results}
  - id: answer
    type: llm_chat
    config: {model: llama3.1:8b, input_key: results, output_key: reply}

edges:                       # routing
  - from: search
    to: answer
  - from: answer
    condition: results==""   # conditional: route on a state expression
    to: fallback
```

## Custom providers

Declare every provider exactly as it is configured in a top-level
`providers:` list. The block is the **single source of truth** — a provider
used by any step's `provider:`, by `default_provider:`, or by `default_model:`
must be declared here, and there is no implicit built-in fallback. Each entry
is a `{name, ...}` mapping that spells out the endpoint:

```yaml
providers:
  - name: vllm
    base_url: http://vllm:8000/v1           # type defaults to openai_compatible
  - name: claude-proxy
    type: anthropic_compatible              # Anthropic wire protocol
    base_url: http://proxy
    chat_path: /v1/messages
    api_key_env: CLAUDE_PROXY_KEY
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    chat_path: /api/chat

steps:
  - id: answer
    type: llm_chat
    config:
      model: meta-llama/Llama-3.3-70B-Instruct
      provider: vllm                        # must be declared in providers:
```

Bare preset-name strings are rejected — every provider is spelled out, so the
file says exactly what is configured. Each `name` must be unique, and only the
recognised provider fields may appear (a stray key is an error). Referencing a
provider name that is not declared here raises `ConfigError`.

The block round-trips through `workflow_to_yaml` / `Flow.to_yaml()`, and the
graph exposes it as `graph.providers` (a `{name: Provider}` map). In code you
can pass the same map straight to `graph.run(state, providers=...)`, which
overrides `graph.providers` for that run.

### Default provider for the whole workflow

A top-level `default_provider:` picks the default for every `llm_chat` /
`react_agent` step that doesn't name one — the YAML equivalent of
`Flow("...", default_provider=...)`:

```yaml
default_provider: ollama
default_model: llama3.1:8b
providers:
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    chat_path: /api/chat
name: chat
steps:
  - id: answer
    type: llm_chat
    config: {}
```

`default_model:` supplies the model for steps that omit their own `model:`
(`LLM(model=...)` still wins). Neither `default_provider` nor `model` /
`default_model` resolved? The step raises `ConfigError`. Steps may still
override the default with their own `provider:` / `model:`.

## The `${ENV}` interpolation

Every value in the document is interpolated against the process environment.
A variable that is not set stays as a literal placeholder — nothing crashes:

```yaml
steps:
  - type: llm_chat
    config:
      api_key_env: ${OPENAI_API_KEY}
```

## Retrying failing steps

Any step can be wrapped with retry logic via a `retry:` block next to its
`config:`.  The block supports ``max_retries`` (attempts, default 3),
``delay`` (seconds between attempts, default 0), ``backoff`` (multiplier
per retry, default 1.0), ``timeout`` (per-attempt timeout), and
``retry_on`` — a list of exception type names or HTTP status codes; by
default every exception is retried.

```yaml
steps:
  - id: search
    type: web_search
    config: {query_key: q, output_key: results}
    retry:
      max_retries: 4
      delay: 0.5
      backoff: 2.0        # delays: 0.5s, 1s, 2s, 4s
      timeout: 30
      retry_on: ["httpx.HTTPStatusError", 429]
```

Use ``retry: {enabled: false}`` to keep the schema valid but disable the
wrapper, and ``retry_on: [429]`` to only retry on that status code.  The
retry wrapper preserves the inner node's normal success/failure behaviour,
so ``__error__`` edges still fire after the final failed attempt.

## Nested subflows (composite agents)

A `subflow` step embeds a complete inner graph — the composite-agent pattern.
The inner graph is declared with the same `steps`/`edges` vocabulary; the outer
graph maps state into it with `input_map` and pulls results back with
`output_map`:

```yaml
steps:
  - id: greet
    type: transform
    config: {action: trim, input_key: text, output_key: text}
  - id: inner
    type: subflow
    config:
      input_map: {text: x}      # outer key → inner key
      output_map: {y: result}   # inner key → outer key
      max_iterations: 50
      graph:                    # the nested graph
        steps:
          - id: up
            type: transform
            config: {action: uppercase, input_key: x, output_key: y}
edges:
  - from: greet
    to: inner
```

Nested graphs validate against the same node registry (any built-in or plugin
type), support `retry:` per inner step, and round-trip through
`workflow_to_yaml`.  Without `input_map`/`output_map` the whole parent state is
passed through.

Alternatively, `config.build` reuses the `agent_step` recipe (context builder →
ReAct harness → append assistant) as a composite agent:

```yaml
steps:
  - id: chat
    type: subflow
    config:
      id_prefix: chat
      build:
        type: agent_step
        system: You are a helpful assistant
        output_key: answer
        model: llama3.1:8b
        messages_key: messages
        use_tools: all
```

## Inspecting a graph

Render the topology back to YAML or as a Mermaid diagram:

```bash
draf graph workflow.yaml          # YAML topology
draf graph workflow.yaml --mermaid # Mermaid flowchart
```

The Mermaid output marks the entry point, annotates edges with their
conditions, and styles ``__error__`` edges distinctly — useful for
docs and review.

## Loading & validating

```python
from draf.yaml import load_workflow
from draf.yaml_schema import validate_workflow_file, format_errors

errors = validate_workflow_file("workflow.yaml")  # [] when ok
if errors:
    print(format_errors(errors, source="workflow.yaml"))

graph, tools, state, reducers = load_workflow("workflow.yaml")
result = await graph.run(state, tools=tools, reducers=reducers)
```

The CLI wraps both:

```bash
draf validate workflow.yaml
draf -f workflow.yaml
```

## Exporting a code-built graph

Build with `Flow`, then serialize to a deployable workflow:

```python
from draf.yaml import workflow_to_yaml, graph_to_yaml

yaml_text = workflow_to_yaml(graph, tools=tools, initial=state, reducers=reducers)
# graph_to_yaml(graph) is a shorthand when you only need the graph itself
```

ReAct edges (`_tool_call_name !=`) round-trip correctly, so an agent built in
code can be emitted as declarative YAML.