# RAG store: qdrant

RAG agent over the Draf knowledge base using the Qdrant store.

## Dependencies

Install the vector-store extra:

```bash
uv add "draf[stores-qdrant]"    # or: pip install "draf[stores-qdrant]"
```

Start a Qdrant server. With Docker Compose (from `examples/rag_stores/`):

```bash
docker compose up -d
```

Or manually:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Ollama must be running locally with:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/rag_stores/run.py examples/rag_stores/qdrant/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py qdrant
```

## Notes

- The client connects to `store.host:store.port` (default
  `localhost:6333`), collection `store.collection` (default `draf`).
- Qdrant is a standalone vector database — designed for larger
  collections, multi-tenant setups, and production scale.
- The `docs.csv` documents are re-uploaded each run (upsert by point id),
  so re-running is idempotent.
