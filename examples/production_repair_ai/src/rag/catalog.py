"""Materials catalog — a small RAG store over ``data/documents/materials.csv``.

Documents are embedded lazily on the first search, so building the catalog
never touches the network; only an actual query requires a configured
embedding provider.
"""

from __future__ import annotations

from draf.rag.stores import InMemoryVectorStore
from draf.rag.tool import load_documents_csv


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
        self._seeded = False

    def add_csv(self, path: str) -> None:
        """Queue a materials CSV for embedding (columns: name, category, ...)."""
        docs = load_documents_csv(path, text_column="description")
        for text, meta in docs:
            record = {**meta, "description": text}
            self._docs.append((text, record))
        self._seeded = False

    async def _ensure_seeded(self) -> None:
        if self._seeded:
            return
        for index, (text, meta) in enumerate(self._docs):
            vector = await self.embedder.embed(text)
            await self.store.add([(f"material_{index}", vector, meta)])
        self._seeded = True

    async def search(
        self,
        query: str,
        category: str | None = None,
        max_price: float | None = None,
        top_k: int | None = None,
    ) -> str:
        """Search the catalog, optionally filtered by category / max price."""
        await self._ensure_seeded()
        query_vector = await self.embedder.embed(query)
        filter_dict = {"category": category} if category else None
        results = await self.store.search(
            query_vector, k=top_k or self.top_k, filter=filter_dict
        )
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
            title = meta.get("name") or _format_record(meta)[:50]
            lines.append(f"{index}. {title} — {price} руб/м² (похожесть: {score:.2f})")
        return "\n".join(lines)


def _parse_price(value) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
