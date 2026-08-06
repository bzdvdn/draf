"""Materials catalog — a small RAG store over ``data/documents/materials.csv``.

Documents are embedded lazily on the first search, so building the catalog
never touches the network; only an actual query requires a configured
embedding provider.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from draf.rag.stores import InMemoryVectorStore
from draf.rag.tool import load_documents_csv


@dataclass
class IngestReport:
    """Outcome of one ingestion run.

    Attributes:
        queued: Documents known to the catalog (parsed, not necessarily embedded).
        added:  Documents newly embedded+stored in this call.
        batches:embedding HTTP calls made this call (``ceil(added / batch_size)``).
        stored: Rows currently resident in the vector store.
    """

    queued: int = 0
    added: int = 0
    batches: int = 0
    stored: int = 0


def _product_text(rec: dict) -> str:
    """Compose the embeddable text for a price-list row.

    *price* is kept as metadata (a number is a poor embedding signal for a
    product); the human-readable identifying columns drive retrieval.
    """
    parts = [
        str(rec[k])
        for k in ("name", "variant", "unit", "brand", "article")
        if rec.get(k)
    ]
    return ", ".join(parts).strip() or (rec.get("name") or "")


def _format_record(meta: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in meta.items() if v)


class MaterialCatalog:
    """Retrieval over the material price list.

    Args:
        embedder: Anything exposing ``async embed(text) -> list[float]``.
        store: :class:`~draf.rag.VectorStore` (defaults to in-memory).
        top_k: Default number of results per search.
    """

    def __init__(self, embedder, store=None, top_k: int = 3):
        self.embedder = embedder
        self.store = store or InMemoryVectorStore(dim=768)
        self.top_k = top_k
        self._docs: list[tuple[str, dict]] = []
        self._ingested = 0

    def add_csv(
        self,
        path: str,
        *,
        text_column: str = "description",
        fieldmap: dict[str, str] | None = None,
    ) -> int:
        """Parse a CSV into the pending queue (no network) and return its size.

        By default the rows are shaped like ``materials.csv``: *text_column*
        becomes the embedded text and the remaining columns become metadata
        (so a ``price`` column must already be named ``price``).

        Pass *fieldmap* (``canonic key -> source column``) to ingest a raw
        price list with Russian column headers — e.g.
        ``{"name": "Наименование", "price": "Цена", "unit": "Ед"}``.  The
        columns are remapped to ``name`` / ``price`` / ``unit`` / ...,
        *name* becomes the embeddable text (rows without a name are
        dropped), and no network is touched until :meth:`ingest`.
        """
        before = len(self._docs)
        if fieldmap is None:
            docs = load_documents_csv(path, text_column=text_column)
            for text, meta in docs:
                record = {**meta, "description": text}
                self._docs.append((text, record))
            return len(self._docs) - before
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                record = {
                    key: (row.get(col, "") or "").strip()
                    for key, col in fieldmap.items()
                }
                name = record.get("name")
                if not name:
                    continue
                text = _product_text(record)
                self._docs.append((text, record))
        return len(self._docs) - before

    @property
    def size(self) -> int:
        """Documents currently known to the catalog (parsed or embedded)."""
        return len(self._docs)

    @property
    def stored(self) -> int:
        """Rows resident in the vector store (embedded)."""
        return self._ingested

    def resume(self) -> None:
        """Adopt rows a durable store already holds (e.g. indexed by a worker).

        A fresh process starts with ``_ingested == 0``, which would re-embed
        everything on the first search.  If the backing store is persistent
        and already populated, treat those rows as embedded so we never
        duplicate work across processes.  Tolerant of stores without a
        synchronous ``count`` (in-memory stores simply stay unchanged).
        """
        sync = getattr(self.store, "count_sync", None)
        if sync is None:
            return
        try:
            self._ingested = min(int(sync()), len(self._docs))
        except ValueError:
            pass

    async def ingest(self, batch_size: int = 250) -> IngestReport:
        """Embed queued-but-not-yet-stored documents into the vector store.

        Documents are embedded in *batch_size* chunks (one ``embed_many``
        call per chunk) and appended to the store.  Safe to call repeatedly:
        already-embedded rows are skipped.  This is the primitive the CLI
        ``load`` command and the catalog API use to pre-fill the store.
        """
        pending = self._docs[self._ingested :]
        batches = 0
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            texts = [text for text, _meta in chunk]
            vectors = await self.embedder.embed_many(texts)
            await self.store.add(
                [
                    (f"material_{self._ingested + start + i}", vectors[i], meta)
                    for i, (_text, meta) in enumerate(chunk)
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
        """Clear the store and re-embed every queued document (full refresh).

        Used by the ``update store`` API/CLI action: drops stale rows, then
        re-indexes the whole catalog so metadata edits or dropped files are
        reflected.
        """
        await self.store.clear()
        self._ingested = 0
        return await self.ingest(batch_size=batch_size)

    async def _ensure_seeded(self) -> None:
        # Lazy fallback: a search against an empty store triggers ingestion,
        # so the graph works even when nobody pre-loaded via CLI/API.
        if self._ingested < len(self._docs):
            await self.ingest()

    async def search(
        self,
        query: str,
        category: str | None = None,
        max_price: float | None = None,
        top_k: int | None = None,
    ) -> str:
        """Search the catalog, optionally filtered by category / max price.

        When a category filter matches nothing (e.g. an LLM passed a room
        type like ``"кухня"`` instead of a catalog category), the search is
        retried without the filter so a useful answer is still returned.
        """
        await self._ensure_seeded()
        query_vector = await self.embedder.embed(query)
        filter_dict = {"category": category} if category else None
        results = await self.store.search(
            query_vector, k=top_k or self.top_k, filter=filter_dict
        )
        if not results and filter_dict:
            results = await self.store.search(query_vector, k=top_k or self.top_k)
        if max_price is not None:
            filtered = []
            for doc_id, score, meta in results:
                price = _parse_price(meta.get("price"))
                if price is None or price <= max_price:
                    filtered.append((doc_id, score, meta))
            results = filtered
        if not results:
            return "Nothing found in materials catalog."
        lines = [
            f"[{index}] {_format_record(meta)}"
            for index, (_, _, meta) in enumerate(results, start=1)
        ]
        return "\n\n".join(lines)

    async def find_similar(self, name: str, top_k: int = 3) -> str:
        """Find materials similar to *name*, reporting similarity scores."""
        await self._ensure_seeded()
        query_vector = await self.embedder.embed(name)
        results = await self.store.search(query_vector, k=top_k)
        if not results:
            return "Nothing found."
        lines = []
        for index, (_, score, meta) in enumerate(results, start=1):
            price = meta.get("price", "-")
            unit = meta.get("unit") or "руб/м²"
            title = meta.get("name") or _format_record(meta)[:50]
            lines.append(f"{index}. {title} — {price} {unit} (похожесть: {score:.2f})")
        return "\n".join(lines)


def _parse_price(value) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
