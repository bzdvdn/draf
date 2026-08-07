# RAG store: lance

RAG agent over the Teff knowledge base using the LanceDB store.

## Dependencies

Install the vector-store extra:

```bash
uv add "teff[stores-lance]"     # or: pip install "teff[stores-lance]"
```

Ollama must be running locally with:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/rag_stores/run.py examples/rag_stores/lance/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py lance
```

## Notes

- LanceDB is an embedded, columnar vector database — no server required.
- Data persists to the `db` directory in this folder.
- Re-running is idempotent: entries are keyed by document chunk id and
  overwritten on re-add.
