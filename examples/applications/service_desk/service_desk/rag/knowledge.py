"""Knowledge base — domain-scoped RAG over the service-desk CSVs.

A single durable :class:`~draf.rag.VectorStore` holds every specialist's
knowledge; each row is tagged with its ``domain`` (incidents / billing /
deploy) so a search is isolated to one specialist via an equality filter.
Documents are embedded lazily on the first search (no network at build
time), and :meth:`resume` adopts rows a durable store already holds so a
restart does not re-embed and duplicate them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from draf.rag.stores import InMemoryVectorStore

DOCUMENT_KEY = "text"


@dataclass
class IngestReport:
    """Outcome of one ingestion run.

    Attributes:
        queued: Documents known to the store (parsed, not necessarily embedded).
        added:  Documents newly embedded+stored in this call.
        batches: Embedding HTTP calls made this call (``ceil(added / batch_size)``).
        stored: Rows resident in the vector store.
    """

    queued: int = 0
    added: int = 0
    batches: int = 0
    stored: int = 0


def _format_record(meta: dict) -> str:
    """Render a result row for the LLM, skipping the domain tag."""
    return ", ".join(f"{k}: {v}" for k, v in meta.items() if v and k != "domain")


class KnowledgeBase:
    """Retrieval over the specialist knowledge CSVs.

    Args:
        embedder: Anything exposing ``async embed(text) -> list[float]`` and
            ``async embed_many(texts) -> list[list[float]]``.
        store: :class:`~draf.rag.VectorStore` (defaults to in-memory).
        top_k: Default number of results per search.
    """

    def __init__(self, embedder, store=None, top_k: int = 3):
        self.embedder = embedder
        self.store = store or InMemoryVectorStore(dim=768)
        self.top_k = top_k
        self._docs: list[tuple[str, dict]] = []
        self._ingested = 0

    def add_csv(self, path: str, *, domain: str, text_column: str = "symptom") -> int:
        """Parse a CSV into the pending queue (no network); rows tagged *domain*."""
        before = len(self._docs)
        with open(path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f), start=1):
                text = (row.get(text_column) or "").strip()
                if not text:
                    continue
                meta = {
                    k: (v or "").strip()
                    for k, v in row.items()
                    if k != text_column and (v or "").strip()
                }
                meta["domain"] = domain
                meta[DOCUMENT_KEY] = text
                self._docs.append((text, meta))
        return len(self._docs) - before

    @property
    def size(self) -> int:
        """Documents known to the base (parsed or embedded)."""
        return len(self._docs)

    @property
    def stored(self) -> int:
        """Rows resident in the vector store (embedded)."""
        return self._ingested

    def resume(self) -> None:
        """Adopt rows a durable store already holds (see ``MaterialCatalog.resume``)."""
        sync = getattr(self.store, "count_sync", None)
        if sync is None:
            return
        try:
            self._ingested = min(int(sync()), len(self._docs))
        except (TypeError, ValueError):
            pass

    async def ingest(self, batch_size: int = 250) -> IngestReport:
        """Embed queued-but-not-yet-stored rows into the vector store."""
        pending = self._docs[self._ingested :]
        batches = 0
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            vectors = await self.embedder.embed_many([t for t, _ in chunk])
            await self.store.add(
                [
                    (f"doc_{self._ingested + start + i}", vectors[i], meta)
                    for i, (_t, meta) in enumerate(chunk)
                ]
            )
            batches += 1
        self._ingested += len(pending)
        return IngestReport(
            queued=len(self._docs),
            added=len(pending),
            batches=batches,
            stored=self._ingested,
        )

    async def rebuild(self, batch_size: int = 250) -> IngestReport:
        """Clear the store and re-embed every queued row (full refresh)."""
        await self.store.clear()
        self._ingested = 0
        return await self.ingest(batch_size=batch_size)

    async def _ensure_seeded(self) -> None:
        """Lazy fallback: a search against an empty store ingests on demand."""
        if self._ingested < len(self._docs):
            await self.ingest()

    async def search(self, query: str, *, domain: str, top_k: int | None = None) -> str:
        """Search within *domain* only, returning formatted rows."""
        await self._ensure_seeded()
        query_vector = await self.embedder.embed(query or "")
        results = await self.store.search(
            query_vector,
            k=top_k or self.top_k,
            filter={"domain": domain},
        )
        if not results:
            return "Ничего не найдено в базе знаний."
        return "\n\n".join(
            f"[{index}] {_format_record(meta)}"
            for index, (_, _, meta) in enumerate(results, start=1)
        )
