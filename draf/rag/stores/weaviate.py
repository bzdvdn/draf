"""Weaviate vector store — requires ``weaviate-client``."""

from __future__ import annotations

import json
import uuid

from draf.rag.base import VectorStore, finalize_results


class WeaviateVectorStore(VectorStore):
    """Vector store backed by Weaviate (cosine, pre-computed vectors).

    Requires the ``weaviate-client`` package. Connect to an existing
    Weaviate instance (``connect_to_local`` by default, ``embedded=True``
    for the embedded server, or pass an HTTP ``host``/``port``/``secure``).
    Metadata is stored as a JSON string property; filters and hybrid scores
    are applied after retrieval.
    """

    def __init__(
        self,
        collection: str = "draf",
        *,
        embedded: bool = False,
        host: str = "localhost",
        http_port: int = 8080,
        http_secure: bool = False,
        grpc_port: int = 50051,
        grpc_secure: bool = False,
        api_key: str = "",
        headers: dict | None = None,
        dim: int | None = None,
    ):
        import weaviate

        self.collection = collection
        self.dim = dim
        if embedded:
            self._client = weaviate.connect_to_embedded(headers=headers)
        else:
            auth = None
            if api_key:
                auth = weaviate.auth.AuthApiKey(api_key)
            self._client = weaviate.connect_to_custom(
                http_host=host,
                http_port=http_port,
                http_secure=http_secure,
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=grpc_secure,
                auth_credentials=auth,
                headers=headers,
            )

    def _ensure_collection(self, dim: int) -> None:
        if self._client.collections.exists(self.collection):
            return
        from weaviate.classes.config import (
            Configure,
            DataType,
            Property,
            StopwordsPreset,
            VectorDistances,
        )

        self._client.collections.create(
            self.collection,
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.flat(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            inverted_index_config=Configure.inverted_index(
                stopwords_preset=StopwordsPreset.NONE
            ),
        )
        self.dim = dim

    @staticmethod
    def _meta_of(properties: dict) -> dict:
        try:
            return json.loads(properties.get("metadata", "{}"))
        except (TypeError, json.JSONDecodeError):
            return {}

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        if not vectors:
            return
        self._ensure_collection(len(vectors[0][1]))
        col = self._client.collections.get(self.collection)
        with col.batch.fixed_size(64) as batch:
            for vid, vec, meta in vectors:
                batch.add_object(
                    uuid=uuid.uuid5(uuid.NAMESPACE_URL, vid),
                    properties={
                        "doc_id": vid,
                        "metadata": json.dumps(meta, ensure_ascii=False),
                    },
                    vector=vec,
                )

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        if not self._client.collections.exists(self.collection):
            return []
        from weaviate.classes.query import MetadataQuery

        col = self._client.collections.get(self.collection)
        n_scan = max(k, k * 4) if (filter or hybrid) else k
        res = col.query.near_vector(
            near_vector=query,
            limit=n_scan,
            return_metadata=MetadataQuery(distance=True),
        )
        candidates = []
        for o in res.objects:
            distance = getattr(o.metadata, "distance", None)
            score = 1.0 - float(distance) if distance is not None else 0.0
            doc_id = o.properties.get("doc_id", str(o.uuid))
            candidates.append((doc_id, score, self._meta_of(o.properties)))
        return finalize_results(candidates, k, filter, hybrid, query_text)

    async def delete(self, ids: list[str]) -> None:
        if not self._client.collections.exists(self.collection):
            return
        from weaviate.classes.query import Filter

        col = self._client.collections.get(self.collection)
        for vid in ids:
            col.data.delete_many(where=Filter.by_property("doc_id").equal(vid))

    async def count(self) -> int:
        if not self._client.collections.exists(self.collection):
            return 0
        col = self._client.collections.get(self.collection)
        return col.aggregate.over_all(total_count=True).total_count

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        if not self._client.collections.exists(self.collection):
            return []
        col = self._client.collections.get(self.collection)
        res = col.query.fetch_objects(limit=limit, offset=offset)
        return [
            (o.properties.get("doc_id", str(o.uuid)), self._meta_of(o.properties))
            for o in res.objects
        ]

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        if not ids or not self._client.collections.exists(self.collection):
            return []
        from weaviate.classes.query import Filter

        col = self._client.collections.get(self.collection)
        cond = Filter.any_of([Filter.by_property("doc_id").equal(vid) for vid in ids])
        res = col.query.fetch_objects(filters=cond, limit=len(ids))
        return [
            (o.properties.get("doc_id", str(o.uuid)), self._meta_of(o.properties))
            for o in res.objects
        ]

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if not self._client.collections.exists(self.collection):
            return
        from weaviate.classes.query import Filter

        col = self._client.collections.get(self.collection)
        res = col.query.fetch_objects(filters=Filter.by_property("doc_id").equal(id))
        if not res.objects:
            return
        obj = res.objects[0]
        merged = {**self._meta_of(obj.properties), **metadata}
        col.data.update(
            uuid=obj.uuid,
            properties={"metadata": json.dumps(merged, ensure_ascii=False)},
        )

    async def clear(self) -> None:
        if self._client.collections.exists(self.collection):
            self._client.collections.delete(self.collection)
