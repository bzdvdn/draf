"""In-memory vector store — zero dependencies."""

from draf.rag.base import VectorStore, cosine_similarity


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

    async def search(self, query: list[float], k: int = 10) -> list[tuple[str, float, dict]]:
        scores = []
        for vid, vec in self._vectors.items():
            sim = cosine_similarity(query, vec)
            scores.append((vid, sim, self._metadatas.get(vid, {})))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    async def delete(self, ids: list[str]) -> None:
        for vid in ids:
            self._vectors.pop(vid, None)
            self._metadatas.pop(vid, None)
