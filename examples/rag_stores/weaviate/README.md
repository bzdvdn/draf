# RAG store: weaviate

RAG agent over the Draf knowledge base using the Weaviate store.

## Dependencies

Install the vector-store extra:

```bash
uv add "draf[embedding]"        # or: pip install "draf[embedding]"
```

Ollama must be running locally with:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/rag_stores/run.py examples/rag_stores/weaviate/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py weaviate
```

## Notes

- This example uses the Weaviate embedded server (`embedded: true`) — no
  external service needed. On first run the embedded Go server binary is
  downloaded automatically.
- To use a running Weaviate instance instead, set `embedded: false` and
  point `host` / `http_port` / `grpc_port` at it.
- Re-running is idempotent: the collection is dropped and recreated on
  each run.
