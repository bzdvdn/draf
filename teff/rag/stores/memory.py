"""In-memory vector store — zero dependencies."""

from __future__ import annotations

from teff.rag.base import VectorStore, cosine_similarity, finalize_results


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using cosine similarity.

    No external dependencies required.  Useful for testing and
    small-scale use cases.

    Args:
        dim: Vector dimensionality (default 1536 for OpenAI ada-002).
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self._vectors: dict[str, list[float]] = {}
        self._metadatas: dict[str, dict] = {}

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        for vid, vec, meta in vectors:
            self._vectors[vid] = vec
            self._metadatas[vid] = meta

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        candidates = [
            (vid, cosine_similarity(query, vec), self._metadatas.get(vid, {}))
            for vid, vec in self._vectors.items()
        ]
        return finalize_results(candidates, k, filter, hybrid, query_text)

    async def delete(self, ids: list[str]) -> None:
        for vid in ids:
            self._vectors.pop(vid, None)
            self._metadatas.pop(vid, None)

    async def count(self) -> int:
        return len(self._vectors)

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        items = sorted(self._vectors)
        return [
            (vid, self._metadatas.get(vid, {}))
            for vid in items[offset : offset + limit]
        ]

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        return [
            (vid, self._metadatas.get(vid, {})) for vid in ids if vid in self._vectors
        ]

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if id in self._vectors:
            self._metadatas[id] = {**self._metadatas.get(id, {}), **metadata}

    async def clear(self) -> None:
        self._vectors.clear()
        self._metadatas.clear()
