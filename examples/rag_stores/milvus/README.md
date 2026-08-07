# RAG store: milvus

RAG agent over the Teff knowledge base using the Milvus store.

## Dependencies

Install the vector-store extra (includes Milvus Lite for local use):

```bash
uv add "teff[stores-milvus]"    # or: pip install "teff[stores-milvus]"
```

Ollama must be running locally with:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/rag_stores/run.py examples/rag_stores/milvus/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py milvus
```

## Notes

- This example uses Milvus Lite: the `uri` is a local file
  (`./milvus.db`), so no Milvus server is required.
- Point the `uri` at a remote server
  (e.g. `http://localhost:19530`) to talk to a full Milvus deployment.
- Re-running is idempotent: the collection is dropped and recreated on
  each run.
