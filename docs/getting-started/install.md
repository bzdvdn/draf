# Install

```bash
pip install draf
```

Python **>=\ 3.11**. Core runtime depends only on `httpx`, `pyyaml`, and
`typer`.

## Extras

Install only what you need; `draf[all]` pulls in everything.

| Extra | Provides |
| ----- | -------- |
| `draf[embedding]` | RAG vector stores (Qdrant, Chroma, FAISS, Lance, Milvus, Weaviate, Pinecone, pgvector) and `asyncpg`/`SQLAlchemy` drivers |
| `draf[rag-pdf]` | `pypdf` — PDF text extraction for RAG |
| `draf[rag-excel]` | `openpyxl` — Excel document loading for RAG |
| `draf[pg-checkpoint]` | `asyncpg` — PostgreSQL checkpoint backend |
| `draf[tools]` | Built-in tools that need third-party deps (web fetch, PDF, S3, Slack, SQL, email, Telegram, …) |
| `draf[mcp]` | Model Context Protocol server tools |
| `draf[fastapi]` | Scaffold templates: `fastapi` + `uvicorn` + `sse-starlette` |
| `draf[queue]` | `celery[redis]` — worker/beat scaffold variant |
| `draf[docs]` | MkDocs toolchain to build this documentation |
| `draf[all]` | Everything above |

```bash
pip install "draf[embedding]"
pip install "draf[tools]"
pip install "draf[all]"
```