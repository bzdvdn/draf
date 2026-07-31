# RAG store: in-memory

RAG agent over the Draf knowledge base using the `in_memory` vector store.

## Dependencies

- Nothing to install — core `draf` (httpx + pyyaml) is enough.
- Ollama must be running locally with:
  - `llama3.1:8b` (chat model)
  - `nomic-embed-text` (embedding model)

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

```bash
# from workflow.yaml (CLI emulation)
uv run python examples/rag_stores/run.py examples/rag_stores/in_memory/workflow.yaml

# from Python Flow API
uv run python examples/rag_stores/flow.py in_memory
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py in_memory
```

## Notes

- Vectors live only in memory — they are lost when the process exits.
- Good for demos, tests, and small workloads. For persistence, use the
  [sqlite](../sqlite) store.
