# Install

```bash
pip install teff
```

Python **>=\ 3.11**. Core runtime depends only on `httpx`, `jsonschema`,
`pyyaml`, and `typer`.

## The CLI

The `teff` CLI ships with the package. Prefer uv? uv installs the package
**and** the CLI in one step:

```bash
uv tool install teff         # global `teff` CLI
uvx teff -f workflow.yaml    # run on the fly without installing anything
```

## Extras

Install only what you need; `teff[all]` pulls in everything.

| Extra | Provides |
| ----- | -------- |
| `teff[stores-qdrant]` | Qdrant vector store (`qdrant-client`) |
| `teff[stores-chroma]` | Chroma vector store (`chromadb`; heavy — pulls `onnxruntime`) |
| `teff[stores-faiss]` | FAISS vector store (`faiss-cpu`) |
| `teff[stores-lance]` | LanceDB vector store (`lancedb`) |
| `teff[stores-milvus]` | Milvus vector store (`pymilvus`, includes Milvus Lite) |
| `teff[stores-weaviate]` | Weaviate vector store (`weaviate-client`) |
| `teff[stores-pinecone]` | Pinecone vector store (`pinecone`) |
| `teff[stores-pgvector]` | PostgreSQL + pgvector store (`asyncpg`, `pgvector`) |
| `teff[embedding]` | Every vector store at once — alias for all `teff[stores-*]` (heaviest) |
| `teff[rag-pdf]` | `pypdf` — PDF text extraction for RAG |
| `teff[rag-excel]` | `openpyxl` — Excel document loading for RAG |
| `teff[pg-checkpoint]` | `asyncpg` — PostgreSQL checkpoint backend |
| `teff[mcp]` | `mcp` — Model Context Protocol tool bridge (`teff.tool.McpTool` / `mcp_tools`, stdio + streamable-http) |
| `teff[tools]` | Built-in tools that need third-party deps (web fetch, PDF, S3, Slack, SQL, email, Telegram, …) |
| `teff[fastapi]` | Scaffold templates: `fastapi` + `uvicorn` + `sse-starlette` |
| `teff[observability]` | Trace dashboard: `fastapi` + `uvicorn` (for `teff.observability.dashboard_router`) |
| `teff[queue]` | `celery[redis]` — worker/beat scaffold variant |
| `teff[docs]` | MkDocs toolchain to build this documentation |
| `teff[all]` | Everything above except `docs` |

```bash
pip install "teff[stores-qdrant]"
pip install "teff[tools]"
pip install "teff[mcp]"
pip install "teff[all]"
```

## Docker

Ready-made images are published to Docker Hub for every `v*` release tag. They
mirror the extras above, so pick the variant that matches your deployment:

| Image                   | Contents                        | Runs                                            |
| ----------------------- | ------------------------------- | ----------------------------------------------- |
| `bzdvdn/teff`           | core + `teff[tools]`            | the `teff` CLI — run/validate/inspect workflows |
| `bzdvdn/teff-fastapi`   | core + `teff[fastapi]`          | `uvicorn` — a FastAPI server app                |
| `bzdvdn/teff-worker`    | core + `teff[queue]`            | `celery` — background workers / beat            |
| `bzdvdn/teff-obs`       | core + `teff[observability]`    | `teff obs-server` — trace dashboard + ingest    |
| `bzdvdn/teff-rag`       | core + `teff[stores-qdrant,tools,rag-pdf]` | the `teff` CLI, slim RAG build          |
| `bzdvdn/teff-all`       | every extra except `docs`       | the `teff` CLI with the full optional surface   |

Run a workflow from a mounted `workflow.yaml` (plus an optional `plugins/`
folder) in one shot:

```bash
docker run --rm -v "$PWD:/workflow" \
  bzdvdn/teff:latest run -f /workflow/workflow.yaml
```

Serve the trace dashboard (ingest + UI) with the collector image:

```bash
docker run -d -p 8001:8001 -v teff-traces:/data \
  bzdvdn/teff-obs:latest --db /data/traces.db --host 0.0.0.0
```

All CLI subcommands and flags work inside the container. Images run as a
non-root user (UID 65534) with durable checkpoints under `/data/checkpoints`
(see the [README](https://github.com/bzdvdn/teff#docker)).