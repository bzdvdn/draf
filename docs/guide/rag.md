# RAG (retrieval)

`RAGTool` chunks, embeds, and retrieves documents from a pluggable vector
store using raw HTTP embeddings. Documents load from CSV, TXT (glob), PDF
(`draf[rag-pdf]`), and Excel (`draf[rag-excel]`). All vector stores are
installed via `draf[embedding]`.

```python
from draf import RAGTool

rag = RAGTool(
    {
        "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
        "store": {"type": "sqlite", "path": "vectors.db", "dim": 768},
        "documents": [
            {"type": "txt", "path": "docs/*.txt"},
            {"type": "csv", "path": "meta.csv", "text_column": "content"},
        ],
        "filter": {"topic": "news"},  # metadata filter (DSL below)
        "similarity_threshold": 0.5,  # drop low-score hits
        "max_tokens": 1024,  # context token budget
        "hybrid": True,  # keyword + semantic blend
    }
)
result = await rag.arun("what changed in v2?")
```

Search args override the config per call: `arun(query, k, filter=...,
similarity_threshold=..., max_tokens=..., parent_retrieval=...)`.

## Embedding providers

All OpenAI-compatible `/v1/embeddings`; `model` is optional (a per-provider
default is used when omitted).

| `provider` | Default `model` | API key env var |
| ---------- | --------------- | --------------- |
| `openai` | `text-embedding-ada-002` | `OPENAI_API_KEY` |
| `ollama` | `nomic-embed-text` | — (local) |
| `mistral` | `mistral-embed` | `MISTRAL_API_KEY` |
| `voyage` | `voyage-3` | `VOYAGE_API_KEY` |
| `jina` | `jina-embeddings-v3` | `JINA_API_KEY` |
| `together` | `togethercomputer/m2-bert-80M-8k-retrieval` | `TOGETHER_API_KEY` |
| `groq` | `nomic-embed-text-v1.5` | `GROQ_API_KEY` |

## Store types

| `type` | Config | Notes |
| --- | --- | --- |
| `in_memory` | `dim` | default; in-process only |
| `sqlite` | `path`, `dim` | stdlib file persistence |
| `chroma` | `path`, `collection` | embedded |
| `qdrant` | `host`, `port`, `collection` | needs a server |
| `pgvector` | `dsn`, `table` | needs PostgreSQL + pgvector |
| `faiss` | `dim`, `path` | FAISS flat index + `.meta.json` sidecar |
| `lance` / `lancedb` | `path`, `table`, `dim` | embedded columnar store |
| `milvus` | `uri`, `token`, `collection`, `dim` | `uri` can be a local `./file.db` (Milvus Lite) |
| `weaviate` | `collection`, `embedded`, `host`, `http_port`, `grpc_port`, `api_key`, `headers`, `dim` | `embedded: true` for the in-process server |
| `pinecone` | `index_name`, `api_key`, `host`, `namespace`, `dim` | API key from `PINECONE_API_KEY` |

## Store management

```python
await store.count()  # number of vectors
await store.entries(limit=100, offset=0)  # (id, metadata) pairs
await store.get(["chunk_0", "chunk_1"])  # by id
await store.update_metadata("chunk_0", {"starred": True})  # merge
await store.clear()  # wipe everything
```

## Features

- **Metadata filters** — `{"category": "news"}` (equality),
  `{"category": ["news", "tech"]}` (membership), and `"$and"` / `"$or"`
  combinators. Honoured by every store.
- **Hybrid search** — blends a lexical keyword score with the embedding score
  (`alpha`, default 0.4) in `InMemoryVectorStore`, `SQLiteVectorStore`, and
  the embedded/external stores.
- **Small-to-big** — with `parent_chunks: true` every chunk keeps its full
  parent text; `parent_retrieval: true` returns whole deduplicated parent
  documents.
- **Token budget** — `max_tokens` truncates the returned context to an
  approximate token count.

## Vision / document agents

Two agent-callable `Tool`s round out the RAG story:

- `PDFTool` — extract text from a PDF page by page (`pypdf`, `draf[rag-pdf]`).
- `ImageTool` — OCR through an OpenAI-compatible vision model.

They are registered with the default tool registry, so a ReAct agent can use
them directly.