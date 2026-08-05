# observability

Inspect what actually goes into every LLM call, across a whole graph run, in
a local web UI — a self-hosted langfuse-style trace viewer built on the
`draf.observability` package.

`POST /api/run` executes a two-LLM graph inside a `GraphObserver`, which
captures:

- the **graph topology** (nodes + edges) for visualisation,
- one **span per visited node** (timing, status, errors),
- every **LLM call with its full request** — the exact `messages` sent to
  the model — and the **response**, tokens, latency, cache hits.

Everything lands in `traces.db` (SQLite). The dashboard at
`GET /obs/ui` (mounted via `draf.observability.dashboard_router`) lists
runs with filters and pagination; clicking a run opens a dedicated page
(`GET /obs/runs/{id}`) with the node list, per-node LLM payloads, prompt
and response side by side, plus editable tags and notes in a side panel.

## Requirements

```
ollama pull llama3.1:8b
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
  -d '{"query": "расскажи про DevOps"}'
```

Open http://localhost:8000/obs/ui and click a run to open its detail page
with the full graph and the exact prompt/response of every model call.

## Where the pieces live

- `draf/observability/model.py` — `Run` / `NodeSpan` / `LLMCall` /
  `GraphTopology` data model (with `from_dict` round-tripping).
- `draf/observability/collector.py` — `GraphObserver`, the wiring between
  `graph.run()` and an exporter (`tracer` + `on_llm_payload` channels).
- `draf/observability/exporter.py` — `TraceExporter` interface,
  `JsonlExporter`, `SQLiteExporter` (also the query layer for the dashboard),
  `CompositeExporter` (fan-out).
- `draf/observability/push.py` — `HttpExporter` (webhook / obs-server ingest),
  `LangfuseExporter`, `LangsmithExporter` — background push, no extra deps.
- `draf/observability/api.py` — `dashboard_router(SQLiteExporter)` and
  `ingest_router(SQLiteExporter)` (`POST /obs/ingest`).
- `draf/observability/builder.py` — `build_observability` /
  `build_observer_factory`: turn a YAML `observability:` block into an
  observer (`draf run` / `draf daemon` use this automatically).
- `draf/observability/server.py` — `build_server` / `serve` (the `draf
  obs-server` command: ingest + dashboard in one process).
- `draf/observability/topology.py` — `topology_from_graph(graph)`.

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
      url: http://localhost:8001/obs/ingest   # another draf obs-server
    - type: langfuse
      host: https://cloud.langfuse.com
      public_key_env: LANGFUSE_PUBLIC_KEY
      secret_key_env: LANGFUSE_SECRET_KEY
```

`draf run -f workflow.yaml` (and `draf daemon`) pick this up automatically;
on the collector side `draf obs-server --db traces.db` serves the same
dashboard.
