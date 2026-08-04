# Install

```bash
pip install draf
```

Python **>=\ 3.11**. Core runtime depends only on `httpx`, `pyyaml`, and
`typer`.

## The CLI

The `draf` CLI ships with the package. Prefer uv? uv installs the package
**and** the CLI in one step:

```bash
uv tool install draf         # global `draf` CLI
uvx draf -f workflow.yaml    # run on the fly without installing anything
```

## Extras

Install only what you need; `draf[all]` pulls in everything.

| Extra | Provides |
| ----- | -------- |
| `draf[stores-qdrant]` | Qdrant vector store (`qdrant-client`) |
| `draf[stores-chroma]` | Chroma vector store (`chromadb`; heavy — pulls `onnxruntime`) |
| `draf[stores-faiss]` | FAISS vector store (`faiss-cpu`) |
| `draf[stores-lance]` | LanceDB vector store (`lancedb`) |
| `draf[stores-milvus]` | Milvus vector store (`pymilvus`, includes Milvus Lite) |
| `draf[stores-weaviate]` | Weaviate vector store (`weaviate-client`) |
| `draf[stores-pinecone]` | Pinecone vector store (`pinecone`) |
| `draf[stores-pgvector]` | PostgreSQL + pgvector store (`asyncpg`, `pgvector`) |
| `draf[embedding]` | Every vector store at once — alias for all `draf[stores-*]` (heaviest) |
| `draf[rag-pdf]` | `pypdf` — PDF text extraction for RAG |
| `draf[rag-excel]` | `openpyxl` — Excel document loading for RAG |
| `draf[pg-checkpoint]` | `asyncpg` — PostgreSQL checkpoint backend |
| `draf[tools]` | Built-in tools that need third-party deps (web fetch, PDF, S3, Slack, SQL, email, Telegram, …) |
| `draf[fastapi]` | Scaffold templates: `fastapi` + `uvicorn` + `sse-starlette` |
| `draf[queue]` | `celery[redis]` — worker/beat scaffold variant |
| `draf[docs]` | MkDocs toolchain to build this documentation |
| `draf[all]` | Everything above except `docs` (MCP tooling is bundled with the core package) |

```bash
pip install "draf[stores-qdrant]"
pip install "draf[tools]"
pip install "draf[all]"
```

## Docker

Ready-made images are published to Docker Hub for every `v*` release tag. They
mirror the extras above, so pick the variant that matches your deployment:

| Image                   | Contents                        | Runs                                            |
| ----------------------- | ------------------------------- | ----------------------------------------------- |
| `bzdvdn/draf`           | core + `draf[tools]`            | the `draf` CLI — run/validate/inspect workflows |
| `bzdvdn/draf-fastapi`   | core + `draf[fastapi]`          | `uvicorn` — a FastAPI server app                |
| `bzdvdn/draf-worker`    | core + `draf[queue]`            | `celery` — background workers / beat            |
| `bzdvdn/draf-rag`       | core + `draf[stores-qdrant,tools,rag-pdf]` | the `draf` CLI, slim RAG build          |
| `bzdvdn/draf-all`       | every extra except `docs`       | the `draf` CLI with the full optional surface   |

Run a workflow from a mounted `workflow.yaml` (plus an optional `plugins/`
folder) in one shot:

```bash
docker run --rm -v "$PWD:/workflow" \
  bzdvdn/draf:latest run -f /workflow/workflow.yaml
```

All CLI subcommands and flags work inside the container. Images run as a
non-root user (UID 65534) with durable checkpoints under `/data/checkpoints`
(see the [README](https://github.com/bzdvdn/draf#docker)).