# Changelog

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
- CLI: `draf new <name>` scaffolding, YAML validation.
