# observability

Inspect what actually goes into every LLM call, across a whole graph run, in
a local web UI — a self-hosted langfuse-style trace viewer built on the
`teff.observability` package.

`POST /api/run` executes a small **`llm -> tools -> llm`** graph inside a
`GraphObserver`, which captures:

- the **graph topology** (nodes + edges) for visualisation,
- one **span per visited node** (timing, status, errors),
- every **LLM call with its full request** — the exact `messages` sent to
  the model — and the **response**, tokens, latency, cache hits,
- every **tool call** (name, arguments, result, ok/error) as a first-class
  entry, parsed out of the LLM message stream.

The graph is a ReAct agent (the `assistant/agent` `react_agent` node ↔ the
`assistant/tool` `tool_exec` node, looping until the model answers) equipped
with two tools — `current_time` and `uppercase` — followed by a final
`summarize` `llm_chat` node that condenses the draft. So the run detail page
shows the whole chain: LLM call → tool call with its result → next LLM call.

Each node span renders an **event timeline**: every LLM call and tool call
gets a numbered step (LLM in indigo, tool in green), connected by a rail, so
you can follow exactly the order the node executed — a ReAct agent with two
tool rounds appears as `1 llm → 2 tool → 3 llm → 4 tool → 5 llm`, each step
expandable to its full prompt/response and tool args/result.

Everything lands in `traces.db` (SQLite). The dashboard at
`GET /obs/ui` (mounted via `teff.observability.dashboard_router`) lists
runs with filters and pagination; clicking a run opens a dedicated page
(`GET /obs/runs/{id}`) with the node list, per-node tool calls and LLM
payloads, prompt and response side by side, plus editable tags and notes in
a side panel.

The same chain is declarative, too — this `workflow.yaml` snippet reproduces
the tool loop with no code (run it with `teff run -f` and it self-traces via
the `observability:` block):

```yaml
providers:
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    chat_path: /api/chat

tools:
  - type: current_time
    config: { provider: ollama }
  - type: calculator
    config: { provider: ollama }

steps:
  - id: assistant
    type: react_agent
    config:
      provider: ollama
      model: qwen2.5:7b
      system: Ты краткий ассистент по DevOps.
      input_key: input
      output_key: draft
      use_tools: [current_time, calculator]
  - id: summarize
    type: llm_chat
    config:
      provider: ollama
      model: qwen2.5:7b
      system: Ты краткий ассистент по DevOps.
      prompt: "Сжато, 1-2 предложения:\n{draft}"
      output_key: answer

observability:
  db: ./traces.db
```

## Requirements

```
ollama pull qwen2.5:7b
uv sync --extra observability   # fastapi + uvicorn for the dashboard
```

## Usage

```
uv run python examples/observability/app.py
```

Then, in another terminal:

```
curl -X POST http://localhost:8000/api/run \
  -H 'Content-Type: application/json' \
  -d '{"query": "Назови текущее время и переведи слово devops в верхний регистр"}'
```

This query actually needs the tools, so the agent calls `current_time` and
`uppercase` and the timeline shows `1 llm → 2 tool → 3 tool → 4 llm`. A
plain question (e.g. `"объясни что такое CI/CD"`) is answered directly and
will show only `llm` steps — the model only calls a tool when the task needs
it, which is the intended agent behaviour. `qwen2.5:7b` is used because it
emits OpenAI-style tool calls reliably; `llama3.1:8b` often answers without
them.

Open http://localhost:8000/obs/ui and click a run to open its detail page
with the full graph and the exact prompt/response of every model call.

## Where the pieces live

- `teff/observability/model.py` — `Run` / `NodeSpan` / `LLMCall` /
  `GraphTopology` data model (with `from_dict` round-tripping).
- `teff/observability/collector.py` — `GraphObserver`, the wiring between
  `graph.run()` and an exporter (`tracer` + `on_llm_payload` channels).
- `teff/observability/exporter.py` — `TraceExporter` interface,
  `JsonlExporter`, `SQLiteExporter` (also the query layer for the dashboard),
  `CompositeExporter` (fan-out).
- `teff/observability/push.py` — `HttpExporter` (webhook / obs-server ingest),
  `LangfuseExporter`, `LangsmithExporter` — background push, no extra deps.
- `teff/observability/api.py` — `dashboard_router(SQLiteExporter)` and
  `ingest_router(SQLiteExporter)` (`POST /obs/ingest`).
- `teff/observability/builder.py` — `build_observability` /
  `build_observer_factory`: turn a YAML `observability:` block into an
  observer (`teff run` / `teff daemon` use this automatically).
- `teff/observability/server.py` — `build_server` / `serve` (the `teff
  obs-server` command: ingest + dashboard in one process).
- `teff/observability/topology.py` — `topology_from_graph(graph)`.

Swap `SQLiteExporter` for `JsonlExporter`, or implement `TraceExporter` to
push runs to langfuse/langsmith — the collector never talks to a concrete
backend.

## Pushing traces without an API

A `workflow.yaml` can trace itself with no code:

```yaml
observability:
  db: ./data/traces.db
  export:
    - type: webhook
      url: http://localhost:8001/obs/ingest   # another teff obs-server
    - type: langfuse
      host: https://cloud.langfuse.com
      public_key_env: LANGFUSE_PUBLIC_KEY
      secret_key_env: LANGFUSE_SECRET_KEY
```

`teff run -f workflow.yaml` (and `teff daemon`) pick this up automatically;
on the collector side `teff obs-server --db traces.db` serves the same
dashboard.
