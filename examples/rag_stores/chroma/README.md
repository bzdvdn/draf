# RAG store: chroma

RAG agent over the Teff knowledge base using the ChromaDB store.

## Dependencies

Install the vector-store extra:

```bash
uv add "teff[stores-chroma]"    # or: pip install "teff[stores-chroma]"
```

For local development against the repo, install directly:

```bash
uv pip install "chromadb>=0.5"
```

Ollama must be running locally with:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/rag_stores/run.py examples/rag_stores/chroma/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py chroma
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py chroma
```

## Notes

- No separate server needed — `chromadb.PersistentClient` writes to a
  local directory (`store.path`, default `./chroma`).
- Enables real ANN search via ChromaDB's HNSW index instead of a brute
  force scan.
- ChromaDB is a heavy dependency (it pulls in `pydantic`, `fastapi`,
  `onnxruntime`, ...); that is why it is an extra rather than a core dep.
