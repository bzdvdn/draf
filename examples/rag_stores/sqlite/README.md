# RAG store: sqlite

RAG agent over the Teff knowledge base using the `sqlite` vector store.

## Dependencies

- Nothing to install — uses Python's stdlib `sqlite3`.
- Ollama must be running locally with:
  - `llama3.1:8b` (chat model)
  - `nomic-embed-text` (embedding model)

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/rag_stores/run.py examples/rag_stores/sqlite/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py sqlite
```

## Notes

- Vectors persist to `./vectors.db` (configurable via `store.path`).
- Search is a brute-force cosine scan over all rows — fine for small to
  medium collections, not for large scale.
- The database file is kept when the process exits, so the store can be
  reused by a later run.
