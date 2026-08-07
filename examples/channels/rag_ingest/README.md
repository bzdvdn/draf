# Channels + RAG ingest: grow a knowledge base from any channel

Send CSV rows (or any text) from Telegram, a webhook, or the terminal; the
workflow normalizes each row through an LLM into one clean paragraph, then
`rag_ingest` chunks, embeds and persists it into a SQLite vector store. A
later `rag` search answers from everything ingested so far.

The write side is a new first-class tool, `rag_ingest`, declared in
`tools:` exactly like `rag` (same `embedder:` / `store:` config). The
AI-parsing step is an explicit `llm_chat` node *before* the tool, so the
pipeline is transparent and the tool stays deterministic.

Requires Ollama: `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`.

## Run

```bash
teff serve examples/channels/rag_ingest/workflow.yaml --port 8000
# or a terminal REPL
teff chat  examples/channels/rag_ingest/workflow.yaml
# or a Telegram bot
TELEGRAM_BOT_TOKEN=... teff bot examples/channels/rag_ingest/workflow.yaml
```

## Ingest over the webhook

```bash
curl -X POST localhost:8000/hooks/ingest \
  -H 'Content-Type: application/json' -H 'X-User-Id: alice' \
  -d '{"row": "sku=A-1|product=Ванна|desc=Чугунная ванна 170 см"}'
# {"ok": true, "session_id": "...", "message": "..."}
```

Each `row` becomes one document: the LLM turns it into a clean paragraph and
`rag_ingest` embeds + stores it in `data/vectors.db`.

## Query the grown knowledge base

Any channel answers from the vectors written above once the base has been
seeded. The store is a plain SQLite file shared by every channel, so
documents ingested over HTTP are searchable from the Telegram bot and the
terminal.

## Tool reference

- `rag_ingest` (write): args `text` / `path` (+ `source_id`, `metadata`);
  config `embedder`, `store`, `type` (csv/txt/pdf/excel), `text_column`,
  `parent_chunks`. See [`teff.rag.RAGIngestTool`](../../../docs/guide/channels.md).
