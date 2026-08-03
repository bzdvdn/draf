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