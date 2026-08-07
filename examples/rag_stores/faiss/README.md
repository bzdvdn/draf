# RAG store: faiss

RAG agent over the Teff knowledge base using the FAISS store.

## Dependencies

Install the vector-store extra:

```bash
uv add "teff[stores-faiss]"     # or: pip install "teff[stores-faiss]"
```

Ollama must be running locally with:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/rag_stores/run.py examples/rag_stores/faiss/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py faiss
```

## Notes

- Vectors are stored in a FAISS flat index file (`index.bin`) plus a
  `index.bin.meta.json` sidecar holding IDs and metadata.
- No server required — everything runs in-process.
- Re-running is idempotent: entries are keyed by document chunk id and
  overwritten on re-add.
