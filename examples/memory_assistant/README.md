# memory_assistant

Long-term memory for an agent, shown several ways:

- `workflow.yaml` — pure YAML, `sqlite` (embedded, no server).
- `workflow.qdrant.yaml` — long-term memory on **Qdrant**.
- `workflow.pgvector.yaml` — long-term memory on **Postgres + pgvector**.
- `main.py` — the same idea in Python: a provider-aware `MemoryStore` +
  `MemoryExtractor` (LLM turns a conversation into durable facts) +
  `memory_context`/`MemoryConfig` injection.

The `memory:` config of the `react_agent` node and the `memory` tool accept
any `VectorStore` type (`in_memory`, `sqlite`, `chroma`, `qdrant`,
`pgvector`, `faiss`, `lance`, `milvus`, `weaviate`, `pinecone`) — the same
stores RAG uses.

## YAML

```bash
export USER_ID=ana
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# embedded (no server)
uv run teff -f examples/memory_assistant/workflow.yaml

# server-backed
docker compose -f examples/memory_assistant/docker-compose.yml up -d
uv run teff -f examples/memory_assistant/workflow.qdrant.yaml
uv run teff -f examples/memory_assistant/workflow.pgvector.yaml
```

Server-backed stores need their extra:
`pip install "teff[stores-qdrant]"` or `"teff[stores-pgvector]"`.

## Python

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
python examples/memory_assistant/main.py
```