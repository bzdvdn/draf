"""Qdrant vector store — requires ``qdrant-client``."""

from __future__ import annotations

import hashlib

from draf.rag.base import VectorStore


def _to_qdrant_filter(filter: dict | None):
    """Translate the Draf filter DSL into a Qdrant ``Filter``."""
    if not filter:
        return None
    from qdrant_client import models

    must: list = []
    should: list = []
    for key, cond in filter.items():
        if key == "$and":
            for sub in cond:
                c = _to_qdrant_filter(sub)
                if c is not None:
                    must.append(c)
        elif key == "$or":
            for sub in cond:
                c = _to_qdrant_filter(sub)
                if c is not None:
                    should.append(c)
        elif isinstance(cond, list):
            must.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=list(cond)))
            )
        else:
            must.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=cond))
            )
    if should:
        return models.Filter(must=must, should=should)
    if must:
        return models.Filter(must=must)
    return None


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant.

    Requires the ``qdrant-client`` package (install via ``draf[embedding]``).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "draf",
        client=None,
    ):
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise ImportError("install qdrant-client for QdrantVectorStore") from e
            client = QdrantClient(host=host, port=port)
        self._client = client
        self._collection = collection

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        collections = {c.name for c in self._client.get_collections().collections}
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    @staticmethod
    def _hash_id(vid: str) -> int:
        return int(hashlib.md5(vid.encode()).hexdigest()[:16], 16)

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        from qdrant_client.models import PointStruct

        if vectors:
            self._ensure_collection(len(vectors[0][1]))
        points = [
            PointStruct(
                id=self._hash_id(vid),
                vector=vec,
                payload={"doc_id": vid, **meta},
            )
            for vid, vec, meta in vectors
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        results = self._client.query_points(
            collection_name=self._collection,
            query=query,
            limit=k,
            query_filter=_to_qdrant_filter(filter),
        )
        return [
            (r.payload.get("doc_id", str(r.id)), r.score, r.payload)  # type: ignore[misc, union-attr]
            for r in results.points
        ]

    async def delete(self, ids: list[str]) -> None:
        hashed = [self._hash_id(vid) for vid in ids]
        self._client.delete(collection_name=self._collection, points_selector=hashed)  # type: ignore[arg-type]

    async def count(self) -> int:
        return self._client.count(collection_name=self._collection).count

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        res = self._client.scroll(
            collection_name=self._collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = res
        return [(p.payload.get("doc_id", str(p.id)), p.payload) for p in points]  # type: ignore[misc, union-attr]

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        hashed = [self._hash_id(vid) for vid in ids]
        res = self._client.retrieve(
            collection_name=self._collection,
            ids=hashed,
            with_payload=True,
            with_vectors=False,
        )
        return [(p.payload.get("doc_id", str(p.id)), p.payload) for p in res]  # type: ignore[misc, union-attr]

    async def update_metadata(self, id: str, metadata: dict) -> None:
        from qdrant_client import models

        self._client.set_payload(
            collection_name=self._collection,
            payload=metadata,
            points=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=id)
                        )
                    ]
                )
            ),
        )

    async def clear(self) -> None:
        from qdrant_client import models

        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[])),
        )
