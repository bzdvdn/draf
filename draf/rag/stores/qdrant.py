"""Qdrant vector store — requires ``qdrant-client``."""

from draf.rag.base import VectorStore


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant.

    Requires the ``qdrant-client`` package (install via ``draf[embedding]``).
    """

    def __init__(self, host: str = "localhost", port: int = 6333, collection: str = "draf"):
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError("install qdrant-client for QdrantVectorStore") from e
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        from qdrant_client.models import PointStruct
        points = []
        for i, (vid, vec, meta) in enumerate(vectors):
            points.append(PointStruct(id=i, vector=vec, payload={"doc_id": vid, **meta}))
        self._client.upsert(collection_name=self._collection, points=points)

    async def search(self, query: list[float], k: int = 10) -> list[tuple[str, float, dict]]:
        results = self._client.search(collection_name=self._collection, query_vector=query, limit=k)
        return [(r.payload.get("doc_id", str(r.id)), r.score, r.payload) for r in results]

    async def delete(self, ids: list[str]) -> None:
        self._client.delete(collection_name=self._collection, points_selector=ids)
