# Changelog

## 0.1.0 — first stable release

Two alphas of development polish went into this line. From here the public
API, YAML surface and CLI are considered stable: breaking changes now
require a minor version bump and a CHANGELOG entry.

Highlights:

- Channels: `teff[channels]` ships HTTP/SSE, Telegram and generic webhook
  adapters bound to one durable `Assistant` built from a `workflow.yaml`: `teff[channels]` ships HTTP/SSE, Telegram and generic webhook
  adapters bound to one durable `Assistant` built from a `workflow.yaml`:
  - `teff serve -f workflow.yaml` runs the HTTP/SSE service; `teff bot`
    runs a Telegram bot (polling or webhook); a `channels:` YAML block
    declares them declaratively.
  - `create_http_app(assistant)` serves `/api/chat` (+ SSE stream, runs
    GET/DELETE) out of the box; `create_http_router(assistant)` returns a
    mountable `APIRouter` so the same endpoints can be embedded in an
    existing FastAPI app. Both accept `dependencies` (auth gates, skipped
    on `/api/health`) and a `turn_kwargs(owner, session_id) -> kwargs`
    hook for per-turn tracing/overrides.
  - One unified turn response `{session_id, waiting, message}` across HTTP,
    webhook and Telegram; owner scoping per channel (`X-User-Id` header,
    Telegram user id, webhook `owner:` spec).
  - `teff.new` scaffolds a `channels` template (YAML-first) and a
    code-first variant.
  - `teff chat` runs the same durable `Assistant` as a terminal REPL —
    interrupts ask in-chat and resume on your answer, so a workflow that
    serves HTTP/Telegram/webhook also works from the shell.
  - `rag_ingest` tool: the write side of the vector store. Declared in
    `tools:` like `rag` (same `embedder:`/`store:` config), it takes
    `text` or a `path` (csv/txt/pdf/excel), chunks, embeds and persists
    documents at runtime — so a Telegram/webhook/terminal turn can grow a
    knowledge base, then answer via `rag`. AI-parsing is an explicit
    `llm_chat` step before the tool.
  - New examples: `examples/channels/supervisor/` (the multi-agent
    supervisor wrapped in the `channels:` block) and
    `examples/channels/rag_ingest/` (ingest CSV rows from any channel).
- `teff.testing.MockLLM` now answers in both OpenAI and Ollama wire shapes,
  so the `mock_llm` fixture works for every provider type.
- `Ask.model(...)` is renamed to `Ask.llm(...)` (the LLM-classifier strategy);
  the internal strategy name is now `"llm"`. The old `model` name clashed with
  the model-name keyword (`Ask.model(model=...)`).
- YAML workflows can now be assembled purely from YAML:
  - `transform` gains pipeline-building actions — `contains`, `compare`
    (numeric `eq/ne/gt/ge/lt/le`), `split`, `join`, `replace`, `coalesce`,
    `pick`, `to_int`, `to_float`, `now`. `contains`/`compare` emit
    `"true"`/`"false"` for direct use in `edges:` conditions.
  - `command` node type: declarative `goto`/`STOP` routing from state
    (`routes:` with `when` conditions + `update:`).
  - `loop` node type: repeat a `body` chain until `state[key]` equals
    `until`, bounded by `max_rounds`.
  - `interrupt` steps accept a `strategy:` shorthand (`equals` / `any_of` /
    `regex` / `llm`) that expands to the classifier + `validate` chain —
    the YAML counterpart of `Flow.interrupt(..., accept=...)`.
  - `include:` block composes steps/edges/tools/state from other workflow
    files (recursive, with optional `prefix:` to avoid id collisions).

## 0.1.0-alpha

First alpha release.

- Workflow as data: `Flow` / `Graph` builders with `route`, `agent_step`,
  `Map`, `Parallel`, `Retry`, `Interrupt` (human-in-the-loop).
- Typed `State` with reducers; YAML workflow definitions.
- Provider layer: Ollama, OpenAI, Anthropic and more, with concurrency control.
- ReAct agent harness, `Tool`/`ToolRegistry`, structured output validation.
- Long-term memory: `MemoryStore`, `MemoryExtractor`, per-owner context.
- RAG: chunker, embedders, vector stores (Chroma, Qdrant, PG, FAISS, ...).
- Observability: run tracing, token pricing, tool-call tracking.
- CLI: `teff new <name>` scaffolding, YAML validation.
