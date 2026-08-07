# YAML workflows

The canonical graph is YAML/JSON — code is optional. Workflows are loaded with
`load_workflow`, validated with `teff validate`, and serialized back out with
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

## Tracing a workflow (`observability:`)

A top-level `observability:` block turns on full-run tracing — topology,
per-node spans and the complete LLM prompt/response — without writing any
code. `teff run` and `teff daemon` pick it up automatically:

```yaml
name: my-workflow

observability:
  db: ./data/traces.db            # local SQLite store (our trace dashboard)
  export:                          # optional: also push to remote sinks
    - type: webhook               # any HTTP endpoint (e.g. our obs-server)
      url: http://obs:8001/obs/ingest
    - type: langfuse              # langfuse public API (Basic auth)
      host: https://cloud.langfuse.com
      public_key_env: LANGFUSE_PUBLIC_KEY
      secret_key_env: LANGFUSE_SECRET_KEY
    - type: langsmith             # langsmith runs API (x-api-key)
      api_key_env: LANGCHAIN_API_KEY
      project: my-project

steps:
  - id: answer
    type: llm_chat
    config: {model: llama3.1:8b, output_key: reply}
```

- `db:` resolves relative to the workflow file; `data/` is created if needed.
- Sinks are fanned out to **all** exporters at once (`CompositeExporter`); a
  failing sink is retried and then logged, never crashes the run.
- Secrets come from environment variables (`*_env`), never from the file.
- Browse the local store in the browser:
  `teff obs-server --db ./data/traces.db --port 8001` → `http://localhost:8001/obs/ui`.
- A remote sink that targets `teff obs-server` needs no API at all — pure YAML
  workflows push their traces over HTTP and the server renders the dashboard.

The same wiring is available in code via
`teff.observability.build_observability` / `build_observer_factory`.

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

## Parallel fan-out

A `parallel` step runs independent branches concurrently and merges their
results back via the state reducers.  Each branch is a single step mapping
_or_ a list of step mappings (run sequentially within the branch):

```yaml
steps:
  - id: fanout
    type: parallel
    config:
      branches:
        - {type: transform, config: {action: uppercase, input_key: q, output_key: web}}
        - [{type: transform, config: {action: count_lines, input_key: q, output_key: upper}},
           {type: transform, config: {action: uppercase, input_key: q, output_key: n}}]
edges:
  - from: fanout
    to: finish
```

Branches receive an isolated copy of the state and may run on the same
or different provider endpoints.  Use `state.schema ... reducer: append`
when branches should accumulate (e.g. collecting `messages`) instead of
overwriting a key.

## Composing workflows (`include:`)

An `include:` block merges steps, edges, tools and state from other workflow
files — recursively, since an included file may include others.  Paths are
relative to the including file:

```yaml
name: composed
include:
  - path: ../shared/workflow.yaml
    prefix: shared_
  - path: ./retry.yaml
```

A `prefix:` (prepended to every included step id and edge endpoint) lets you
compose the same file into several places without id collisions; it is also
applied to `command` node `goto` targets.  Included steps run first, then the
including workflow's own steps.  Without a prefix, ids must not collide.

## Declarative routing (`command`)

A `command` step routes the graph from state — dynamic `goto` / `STOP`
without code.  The `when` expressions use the same language as `edges:`
conditions, and the first match wins:

```yaml
steps:
  - id: route
    type: command
    config:
      routes:
        - {when: "score >= 0.8", goto: approve}
        - {when: "score < 0.3", goto: reject}
      goto: review
      update: {routed: true}
```

`goto: STOP` terminates the run.

## Loops (`loop`)

A `loop` step repeats a `body` chain until `state[key]` equals `until` —
everything in one node, no decider edges.  `body` is a node or list of nodes
given as inline `type: ...` specs (like `map`'s processor):

```yaml
steps:
  - id: refine
    type: loop
    config:
      key: approved
      until: "да"
      max_rounds: 3
      body:
        - {type: transform, config: {action: value, value: "нет", output_key: approved}}
```

`max_rounds` (default 10) bounds the repetition; the condition uses the edges
expression language, so `until: "да"` matches `"Да"` or `"да."`.

## Validated interrupts (`strategy:`)

An `interrupt` step can validate the operator's answer with a `strategy:`
mapping instead of comparing it verbatim.  The loader expands it into the
classifier + `validate` chain (`{id}-validate`), the YAML counterpart of
`flow.interrupt(key, prompt, accept=...)`:

```yaml
steps:
  - id: gate
    type: interrupt
    config:
      key: approved
      prompt: "Approve the report? (yes / no)"
      strategy: {equals: да}          # or: any_of: [да, ок] | regex: "^[A-Z0-9]{4}$"
  - id: ship
    type: transform
    config: {action: value, value: shipped, output_key: status}
edges:
  - {from: gate, to: ship, condition: "decision=да"}
```

An `llm` strategy needs `model` and `provider`:

```yaml
      strategy:
        llm:
          system: Classify the answer as approval or rejection.
          user: "Answer: {approved}"
          schema:
            type: object
            properties: {ok: {type: boolean}}
          model: llama3.1:8b
          provider: ollama
```

Edges that would have sourced from the interrupt now source from
`{id}-validate`, where the decision key (`decision` by default) is written.

## Durable runs (`checkpoint:`)

A top-level `checkpoint:` block enables durable runs whose state is saved
before every node, so an interrupted or crashed workflow resumes instead of
restarting:

```yaml
checkpoint:
  type: sqlite              # file | sqlite | sqlite_history | pg | pg_history
  path: data/checkpoints.db
```

`path` is resolved relative to the workflow file.  PG variants require
`dsn:` (+ optional `table:`).  Use `teff run --checkpoint-id <id>` (or the
`--checkpoint '...'` JSON flag to override the block per invocation).  This
is the same durable machinery the `teff` CLI and the conversational
turn/`Assistant` layer use.

## Hook events (`hooks:`)

Hooks observe node execution. Because they're Python callbacks, a workflow
_names_ hooks registered in a plugin — declare the plugin under `plugins:`,
register them with the `@hooks.hook` decorator, then reference by name:

```python
# plugins/telemetry.py
from teff import hooks


@hooks.hook("tick")
def tick(node_id, node, state, **kwargs):
    metrics.counter("graph.node", node_id=node_id, type=node.type)
```

```yaml
plugins: [plugins/telemetry.py]
hooks:
  on_node_start: tick          # (node_id, node, state)
  on_node_end: [tick, finalize] # also passes the node result
  on_node_error: on_error        # also passes the exception
```

Each key takes a hook-name string or a list; an unknown name fails
validation with a clear message.  Sync and async hooks are both supported
(`graph.run` awaits async ones).  The same `hooks=` mapping can be passed
programmatically.

## Inspecting a graph

Render the topology back to YAML or as a Mermaid diagram:

```bash
teff graph workflow.yaml          # YAML topology
teff graph workflow.yaml --mermaid # Mermaid flowchart
```

The Mermaid output marks the entry point, annotates edges with their
conditions, and styles ``__error__`` edges distinctly — useful for
docs and review.

## Loading & validating

```python
from teff.yaml import load_workflow
from teff.yaml_schema import validate_workflow_file, format_errors

errors = validate_workflow_file("workflow.yaml")  # [] when ok
if errors:
    print(format_errors(errors, source="workflow.yaml"))

graph, tools, state, reducers = load_workflow("workflow.yaml")
result = await graph.run(state, tools=tools, reducers=reducers)
```

The CLI wraps both:

```bash
teff validate workflow.yaml
teff -f workflow.yaml
```

## Exporting a code-built graph

Build with `Flow`, then serialize to a deployable workflow:

```python
from teff.yaml import workflow_to_yaml, graph_to_yaml

yaml_text = workflow_to_yaml(graph, tools=tools, initial=state, reducers=reducers)
# graph_to_yaml(graph) is a shorthand when you only need the graph itself
```

ReAct edges (`_tool_call_name !=`) round-trip correctly, so an agent built in
code can be emitted as declarative YAML.