# RAG store: pgvector

RAG agent over the Draf knowledge base using the PostgreSQL + pgvector
store.

## Dependencies

Install the vector-store extra:

```bash
uv add "draf[embedding]"        # or: pip install "draf[embedding]"
```

Start PostgreSQL with the pgvector extension. With Docker Compose (from
`examples/rag_stores/`):

```bash
docker compose up -d
```

Or manually:

```bash
docker run -d \
  -e POSTGRES_PASSWORD=postgres \
  -p 5433:5432 \
  pgvector/pgvector:pg16
```

Create the vector extension and the target table (the example assumes
table `draf_vectors`):

```bash
docker exec -it <container> psql -U postgres -c 'CREATE EXTENSION IF NOT EXISTS vector'
docker exec -it <container> psql -U postgres -c \
  'CREATE TABLE IF NOT EXISTS draf_vectors (doc_id text, embedding vector, metadata jsonb)'
```

Ollama must be running locally with:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

From `workflow.yaml` (CLI emulation):

```bash
uv run python examples/rag_stores/run.py examples/rag_stores/pgvector/workflow.yaml
```

From code with the Python Flow API:

```bash
uv run python examples/rag_stores/flow.py pgvector
```

## Notes

- Connection string is `store.dsn` (default in the example:
  `postgresql://postgres:postgres@localhost:5433/postgres`); adjust user /
  password / host as needed. The compose file maps the container's 5432 to
  the host's 5433 so a locally running PostgreSQL is not disturbed.
- Uses pgvector's `<=>` (cosine distance) ANN index for search.
- Best when you already run PostgreSQL and want vectors in the same
  database as the rest of your data.
