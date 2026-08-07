"""Write tool — add documents to a vector store from a workflow YAML.

``rag`` searches an existing store; ``rag_ingest`` is the write side: it
takes document text or a file (csv / txt / pdf / excel), chunks it, embeds
it and persists the vectors in the same :class:`~teff.rag.base.VectorStore`
config format.  Together they turn a vector store into a living knowledge
base that a workflow can grow at runtime — e.g. a Telegram user drops a CSV
and the agent embeds it before answering.

AI-parsing happens *before* the tool in the workflow: run an ``llm_chat``
(or ``transform``) node that normalizes the raw payload into clean text and
write it to a state key, then ``rag_ingest`` with ``text: "{that_key}"``.
The tool itself only loads, chunks, embeds and stores — no hidden model
calls, so it is deterministic and cheap to test.
"""

from __future__ import annotations

from teff.rag.chunker import Chunker
from teff.rag.embedder import embedder_from_config
from teff.rag.tool import _DOCUMENT_LOADERS
from teff.tool.tool import Tool


class RAGIngestTool(Tool):
    """Add documents to a vector store.

    Args:
        text: Raw document text to chunk, embed and store (inline content).
        path: File to load instead of *text* (see config ``type``).
        source_id: Optional stable id for the document (default: derived).
        metadata: Extra metadata dict merged into every chunk.

    Args (config):
        embedder: Embedder config (same shape as the ``rag`` tool).
        store: Vector-store config (same shape as the ``rag`` tool).
        chunker: Optional chunker kwargs.
        type: Loader for ``path`` — ``csv`` (default), ``txt``, ``pdf``,
            ``excel``.  Ignored when ``text`` is provided.
        text_column: Column used as text when loading a CSV/Excel file.
        parent_chunks: Keep full parent text per chunk (default false).

    At least one of ``text`` or ``path`` must be supplied per call.  The
    result is a short confirmation with the number of chunks written.
    """

    name = "rag_ingest"
    description = (
        "Add a document to the knowledge base: give 'text' (content) or "
        "'path' (a csv/txt/pdf/excel file); it is chunked, embedded and "
        "stored for later 'rag' searches."
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.embedder = embedder_from_config(cfg)

        from teff.rag.stores.factory import store_from_config

        self.store = store_from_config(cfg.get("store") or {})
        self.chunker = Chunker(**(cfg.get("chunker") or {}))
        self.loader_type = cfg.get("type", "csv")
        self.text_column = cfg.get("text_column", "text")
        self.parent_chunks = bool(cfg.get("parent_chunks", False))

    async def arun(  # type: ignore[override]
        self,
        text: str = "",
        path: str = "",
        source_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        docs: list[tuple[str, dict]] = []
        if text.strip():
            docs.append((text, {"id": source_id or "inline"}))
        elif path:
            loader = _DOCUMENT_LOADERS.get(self.loader_type)
            if loader is None:
                msg = f"unsupported document type: {self.loader_type}"
                raise ValueError(msg)
            kwargs: dict = {"path": path}
            if self.loader_type in ("csv", "excel"):
                kwargs["text_column"] = self.text_column
            docs = loader(**kwargs)
            if not docs:
                return f"no documents loaded from {path}"
        else:
            raise ValueError("rag_ingest requires 'text' or 'path'")

        extra = metadata or {}
        total = 0
        for doc_text, doc_meta in docs:
            meta = {**doc_meta, **extra}
            if source_id:
                meta["id"] = source_id
            await self._store(doc_text, meta)
            total += 1

        return f"ingested {total} document(s), {len(docs)} chunked+embedded"

    async def _store(self, text: str, metadata: dict) -> None:
        import uuid

        chunks = self.chunker.chunk(text)
        parent_id = metadata.get("id") or f"doc_{uuid.uuid4().hex[:8]}"
        embeddings = await self.embedder.embed_many(chunks)
        vectors = []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            if self.parent_chunks:
                doc_id = f"{parent_id}_{i}"
                meta = {
                    **metadata,
                    "id": parent_id,
                    "parent_id": parent_id,
                    "parent_text": text,
                    "text": chunk,
                    "chunk_index": i,
                }
            else:
                doc_id = f"{parent_id}_{i}"
                meta = {**metadata, "text": chunk, "chunk_index": i}
            vectors.append((doc_id, vec, meta))
        await self.store.add(vectors)


__all__ = ["RAGIngestTool"]
