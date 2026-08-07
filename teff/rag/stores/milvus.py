"""Milvus vector store — requires ``pymilvus``."""

from __future__ import annotations

import json

from teff.rag.base import VectorStore, finalize_results


def _to_milvus_expr(filter: dict | None) -> str:
    """Translate the Teff filter DSL into a Milvus boolean expression."""

    def build(node: dict) -> str:
        parts: list[str] = []
        for key, cond in node.items():
            if key in ("$and", "$or"):
                op = " and " if key == "$and" else " or "
                subs = [build(c) for c in cond]
                subs = [s for s in subs if s]
                if subs:
                    parts.append("(" + op.join(subs) + ")")
            elif isinstance(cond, list):
                vals = ", ".join(json.dumps(c) for c in cond)
                parts.append(f"{key} in [{vals}]")
            else:
                parts.append(f"{key} == {json.dumps(cond)}")
        return " and ".join(parts)

    return build(filter) if filter else ""


class MilvusVectorStore(VectorStore):
    """Vector store backed by Milvus.

    Requires the ``pymilvus`` package. Uses cosine distance (scores are
    similarity-like, higher = more similar). Works against a Milvus server
    or a local Milvus Lite file when the ``uri`` is a local path
    (``pymilvus[milvus_lite]``).
    """

    def __init__(
        self,
        uri: str = "./milvus.db",
        token: str = "",
        collection: str = "teff",
        dim: int | None = None,
    ):
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=uri, token=token or None)
        self.collection = collection
        self.dim = dim
        self._created = False

    def _ensure_collection(self, dim: int) -> None:
        if self._created:
            return
        from pymilvus import DataType, MilvusClient

        if self._client.has_collection(self.collection):
            self._created = True
            return
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            "id", datatype=DataType.VARCHAR, is_primary=True, max_length=512
        )
        schema.add_field("vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )
        self._client.create_collection(
            self.collection, schema=schema, index_params=index_params
        )
        self.dim = dim
        self._created = True

    @staticmethod
    def _meta_from(entity: dict) -> dict:
        return {k: v for k, v in entity.items() if k not in ("id", "vector")}

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        if not vectors:
            return
        self._ensure_collection(len(vectors[0][1]))
        rows = [{"id": vid, "vector": vec, **meta} for vid, vec, meta in vectors]
        self._client.insert(self.collection, rows)

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        if not self._client.has_collection(self.collection):
            return []
        limit = max(k, k * 4) if hybrid else k
        res = self._client.search(
            self.collection,
            data=[query],
            limit=limit,
            output_fields=["*"],
            filter=_to_milvus_expr(filter),
        )
        if not res:
            return []
        candidates = [
            (hit["id"], float(hit["distance"]), self._meta_from(hit.get("entity", {})))
            for hit in res[0]
        ]
        return finalize_results(
            candidates, k, filter=None, hybrid=hybrid, query_text=query_text
        )

    async def delete(self, ids: list[str]) -> None:
        if not self._client.has_collection(self.collection):
            return
        vals = ", ".join(json.dumps(i) for i in ids)
        self._client.delete(self.collection, filter=f"id in [{vals}]")

    async def count(self) -> int:
        if not self._client.has_collection(self.collection):
            return 0
        return int(self._client.get_collection_stats(self.collection)["row_count"])

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        if not self._client.has_collection(self.collection):
            return []
        res = self._client.query(
            self.collection, filter="", output_fields=["*"], limit=limit, offset=offset
        )
        return [(row["id"], self._meta_from(dict(row))) for row in res]

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        if not ids or not self._client.has_collection(self.collection):
            return []
        vals = ", ".join(json.dumps(i) for i in ids)
        res = self._client.query(
            self.collection, filter=f"id in [{vals}]", output_fields=["*"]
        )
        return [(row["id"], self._meta_from(dict(row))) for row in res]

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if not self._client.has_collection(self.collection):
            return
        res = self._client.query(
            self.collection, filter=f"id == {json.dumps(id)}", output_fields=["*"]
        )
        rows = list(res)
        if not rows:
            return
        row = dict(rows[0])
        vector = row.get("vector")
        merged = {**self._meta_from(row), **metadata}
        self._client.upsert(self.collection, [{"id": id, "vector": vector, **merged}])

    async def clear(self) -> None:
        if self._client.has_collection(self.collection):
            self._client.drop_collection(self.collection)
            self._created = False
