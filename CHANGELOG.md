# Changelog

## Unreleased

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
