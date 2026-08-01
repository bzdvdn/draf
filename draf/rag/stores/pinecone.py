"""Pinecone vector store — requires ``pinecone``."""

from __future__ import annotations

import os

from draf.rag.base import VectorStore, finalize_results


def _to_pinecone_filter(filter: dict | None) -> dict | None:
    """Translate the Draf filter DSL into a Pinecone metadata filter."""
    if not filter:
        return None
    out: dict = {}
    for key, cond in filter.items():
        if key in ("$and", "$or"):
            out[key] = [_to_pinecone_filter(c) for c in cond]
        elif isinstance(cond, list):
            out[key] = {"$in": list(cond)}
        else:
            out[key] = cond
    return out


class PineconeVectorStore(VectorStore):
    """Vector store backed by Pinecone (managed, cloud).

    Requires the ``pinecone`` package and an existing index (created in
    the Pinecone console/API). The API key is read from ``PINECONE_API_KEY``
    or passed explicitly. Scores are cosine similarity from the index.
    """

    def __init__(
        self,
        index_name: str = "draf",
        api_key: str = "",
        host: str = "",
        namespace: str = "",
        dim: int | None = None,
    ):
        from pinecone import Pinecone

        key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not key:
            msg = "Pinecone API key not found (set PINECONE_API_KEY or pass api_key)"
            raise ValueError(msg)
        self._pc = Pinecone(api_key=key)
        self.index_name = index_name
        self.namespace = namespace
        self.dim = dim
        self._index = (
            self._pc.Index(index_name, host=host)
            if host
            else self._pc.Index(index_name)
        )

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        if not vectors:
            return
        self._index.upsert(
            vectors=[(vid, vec, meta) for vid, vec, meta in vectors],
            namespace=self.namespace,
        )

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        top_k = max(k, k * 4) if (filter or hybrid) else k
        res = self._index.query(
            vector=query,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=_to_pinecone_filter(filter),
        )
        candidates = [(m.id, float(m.score), m.metadata or {}) for m in res.matches]
        return finalize_results(
            candidates, k, filter=None, hybrid=hybrid, query_text=query_text
        )

    async def delete(self, ids: list[str]) -> None:
        self._index.delete(ids=ids, namespace=self.namespace)

    async def count(self) -> int:
        return int(self._index.describe_index_stats().total_vector_count)

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        if not ids:
            return []
        res = self._index.fetch(ids=ids, namespace=self.namespace)
        return [(vid, rec.metadata or {}) for vid, rec in (res.vectors or {}).items()]

    async def update_metadata(self, id: str, metadata: dict) -> None:
        self._index.update(id=id, set_metadata=metadata, namespace=self.namespace)

    async def clear(self) -> None:
        self._index.delete(delete_all=True, namespace=self.namespace)
