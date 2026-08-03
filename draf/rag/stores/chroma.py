"""Chroma vector store — requires ``chromadb``."""

from __future__ import annotations

from draf.rag.base import VectorStore


def _to_chroma_where(filter: dict | None) -> dict | None:
    """Translate the Draf filter DSL into a Chroma ``where`` clause."""
    if not filter:
        return None
    out: dict = {}
    for key, cond in filter.items():
        if key in ("$and", "$or"):
            subs = [_to_chroma_where(c) for c in cond]
            subs = [s for s in subs if s]
            if subs:
                out[key] = subs
        elif isinstance(cond, list):
            out[key] = {"$in": list(cond)}
        else:
            out[key] = {"$eq": cond}
    return out


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB (persistent, cosine space).

    Requires the ``chromadb`` package (install via ``draf[embedding]``).
    The collection is created with cosine distance; search returns
    similarity scores (``1 - cosine_distance``).
    """

    def __init__(self, path: str = "./chroma", collection: str = "draf"):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("install chromadb for ChromaVectorStore") from e
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        ids = [v[0] for v in vectors]
        embeddings = [v[1] for v in vectors]
        metadatas = [v[2] for v in vectors]
        self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)  # type: ignore[arg-type]

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        results = self._collection.query(
            query_embeddings=[query],  # type: ignore[arg-type]
            n_results=k,
            where=_to_chroma_where(filter),
            include=["metadatas", "distances"],
        )
        ids = results.get("ids") or [[]]
        distances = results.get("distances") or [[]]
        metadatas = results.get("metadatas") or [[]]
        out = []
        for i in range(len(ids[0])):
            score = 1.0 - distances[0][i]
            meta = (metadatas[0][i] or {}) if metadatas[0] else {}
            out.append((ids[0][i], score, meta))
        return out  # type: ignore[return-value]

    async def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    async def count(self) -> int:
        return self._collection.count()

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        res = self._collection.get(limit=limit, offset=offset, include=["metadatas"])
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        return [(ids[i], metas[i] or {}) for i in range(len(ids))]  # type: ignore[misc]

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        res = self._collection.get(ids=ids, include=["metadatas"])
        got = res.get("ids") or []
        metas = res.get("metadatas") or []
        return [(got[i], metas[i] or {}) for i in range(len(got))]  # type: ignore[misc]

    async def update_metadata(self, id: str, metadata: dict) -> None:
        res = self._collection.get(ids=[id], include=["metadatas"])
        if not (res.get("ids") or []):
            return
        current = (res.get("metadatas") or [{}])[0] or {}
        self._collection.update(ids=[id], metadatas=[{**current, **metadata}])

    async def clear(self) -> None:
        res = self._collection.get(include=[])
        ids = res.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
