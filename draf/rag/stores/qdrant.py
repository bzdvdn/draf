"""Qdrant vector store — requires ``qdrant-client``."""

import hashlib

from draf.rag.base import VectorStore


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant.

    Requires the ``qdrant-client`` package (install via ``draf[embedding]``).
    """

    def __init__(
        self, host: str = "localhost", port: int = 6333, collection: str = "draf"
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError("install qdrant-client for QdrantVectorStore") from e
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        collections = {c.name for c in self._client.get_collections().collections}
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        from qdrant_client.models import PointStruct

        if vectors:
            self._ensure_collection(len(vectors[0][1]))
        points = [
            PointStruct(
                id=int(hashlib.md5(vid.encode()).hexdigest()[:16], 16),
                vector=vec,
                payload={"doc_id": vid, **meta},
            )
            for vid, vec, meta in vectors
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self, query: list[float], k: int = 10
    ) -> list[tuple[str, float, dict]]:
        results = self._client.query_points(
            collection_name=self._collection, query=query, limit=k
        )
        return [
            (r.payload.get("doc_id", str(r.id)), r.score, r.payload)
            for r in results.points
        ]

    async def delete(self, ids: list[str]) -> None:
        self._client.delete(collection_name=self._collection, points_selector=ids)
