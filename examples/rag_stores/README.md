# RAG store examples

The same RAG agent (docs.csv → embeddings → search → answer) running on
every vector store draf supports. Pick the one that fits your needs.

| Store       | Deps needed                | Server required | Persistence    | Best for                          |
| ----------- | -------------------------- | --------------- | -------------- | --------------------------------- |
| `in_memory` | none (core)                | no              | no             | demos, tests, small workloads     |
| `sqlite`    | none (stdlib `sqlite3`)    | no              | yes (file)     | small collections, file storage   |
| `chroma`    | `draf[embedding]`          | no (local dir)  | yes (directory)| local ANN search, no server       |
| `faiss`     | `draf[embedding]`          | no              | yes (files)    | fast flat index, in-process       |
| `lance`     | `draf[embedding]`          | no              | yes (directory)| embedded columnar store           |
| `milvus`    | `draf[embedding]`          | no (Milvus Lite)| yes (file)     | local/remote Milvus, scaling path  |
| `weaviate`  | `draf[embedding]`          | no (embedded)   | in-memory      | local ANN search, embedded server |
| `qdrant`    | `draf[embedding]`          | yes (Docker)    | yes            | production scale, multi-tenant    |
| `pgvector`  | `draf[embedding]`          | yes (PostgreSQL)| yes            | vectors alongside your SQL data   |
| `pinecone`  | `draf[embedding]`          | yes (cloud)     | yes            | managed, hosted vector search     |

Each subdirectory has its own README with the exact install steps:

- [in_memory](in_memory/README.md)
- [sqlite](sqlite/README.md)
- [chroma](chroma/README.md)
- [faiss](faiss/README.md)
- [lance](lance/README.md)
- [milvus](milvus/README.md)
- [weaviate](weaviate/README.md)
- [qdrant](qdrant/README.md)
- [pgvector](pgvector/README.md)

## Shared prerequisites

All examples need Ollama running locally with the chat and embedding
models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

The vector-store extra (for `chroma`, `qdrant`, `pgvector`):

```bash
uv add "draf[embedding]"        # or: pip install "draf[embedding]"
```

The stores that need a server (`qdrant`, `pgvector`) ship a
`docker-compose.yml` — start them all from `examples/rag_stores/`:

```bash
docker compose up -d
```

## Run any of them

Two entry points — same agent, same documents:

`run.py` loads the workflow from `workflow.yaml` (CLI emulation):

```bash
uv run python examples/rag_stores/run.py examples/rag_stores/in_memory/workflow.yaml
uv run python examples/rag_stores/run.py examples/rag_stores/sqlite/workflow.yaml
uv run python examples/rag_stores/run.py examples/rag_stores/chroma/workflow.yaml
uv run python examples/rag_stores/run.py examples/rag_stores/faiss/workflow.yaml
uv run python examples/rag_stores/run.py examples/rag_stores/lance/workflow.yaml
uv run python examples/rag_stores/run.py examples/rag_stores/milvus/workflow.yaml
uv run python examples/rag_stores/run.py examples/rag_stores/weaviate/workflow.yaml
uv run python examples/rag_stores/run.py examples/rag_stores/qdrant/workflow.yaml
uv run python examples/rag_stores/run.py examples/rag_stores/pgvector/workflow.yaml
```

`flow.py` builds the same agent with the Python Flow API (pass the store
type as an argument):

```bash
uv run python examples/rag_stores/flow.py in_memory
uv run python examples/rag_stores/flow.py sqlite
uv run python examples/rag_stores/flow.py chroma
uv run python examples/rag_stores/flow.py faiss
uv run python examples/rag_stores/flow.py lance
uv run python examples/rag_stores/flow.py milvus
uv run python examples/rag_stores/flow.py weaviate
uv run python examples/rag_stores/flow.py qdrant
uv run python examples/rag_stores/flow.py pgvector
uv run python examples/rag_stores/flow.py pinecone
```

`pinecone` has no subdirectory: it needs a hosted index, so configure it
in `flow.py` and set `PINECONE_API_KEY` (plus create an index named
`draf`) before running.
