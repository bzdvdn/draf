"""FAISS vector store — requires ``faiss-cpu``."""

from __future__ import annotations

import json
import os

from draf.rag.base import VectorStore, finalize_results


class FAISSVectorStore(VectorStore):
    """Vector store backed by FAISS (flat inner-product index).

    Requires the ``faiss-cpu`` package. Vectors are L2-normalized on add,
    so inner-product scores equal cosine similarity. Persistence: a FAISS
    ``.index`` file plus a ``.meta.json`` sidecar holding IDs and metadata.
    """

    def __init__(self, dim: int = 1536, path: str | None = None):
        import faiss

        self.dim = dim
        self.path = path
        self._vectors: dict[str, list] = {}
        self._metadatas: dict[str, dict] = {}
        self._index = faiss.IndexFlatIP(dim)
        if path:
            if os.path.exists(path):
                self._index = faiss.read_index(path)
            meta_path = path + ".meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._vectors = {
                    k: self._normalized(v)
                    for k, v in (data.get("vectors") or {}).items()
                }
                self._metadatas = data.get("metadatas", {})

    @staticmethod
    def _normalized(vec: list[float]) -> list:
        import numpy as np

        a = np.asarray(vec, dtype="float32")
        n = float(np.linalg.norm(a))
        return (a / n).tolist() if n > 0 else a.tolist()

    def _rebuild(self) -> None:
        import faiss
        import numpy as np

        idx = faiss.IndexFlatIP(self.dim)
        if self._vectors:
            idx.add(np.vstack(list(self._vectors.values())).astype("float32"))
        self._index = idx

    def _persist(self) -> None:
        if not self.path:
            return
        import faiss

        faiss.write_index(self._index, self.path)
        with open(self.path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {"vectors": self._vectors, "metadatas": self._metadatas},
                f,
                ensure_ascii=False,
            )

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        for vid, vec, meta in vectors:
            if len(vec) != self.dim:
                msg = f"vector for '{vid}' has dim {len(vec)}, expected {self.dim}"
                raise ValueError(msg)
            self._vectors[vid] = self._normalized(vec)
            self._metadatas[vid] = meta
        self._rebuild()
        self._persist()

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        if not self._vectors:
            return []
        import numpy as np

        n_scan = max(k, k * 4) if (filter or hybrid) else k
        n_scan = min(n_scan, self._index.ntotal)
        q = np.asarray(self._normalized(query), dtype="float32").reshape(1, -1)
        scores, positions = self._index.search(q, n_scan)
        ids = list(self._vectors.keys())
        candidates = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0 or pos >= len(ids):
                continue
            vid = ids[pos]
            candidates.append((vid, float(score), self._metadatas.get(vid, {})))
        return finalize_results(candidates, k, filter, hybrid, query_text)

    async def delete(self, ids: list[str]) -> None:
        for vid in ids:
            self._vectors.pop(vid, None)
            self._metadatas.pop(vid, None)
        self._rebuild()
        self._persist()

    async def count(self) -> int:
        return len(self._vectors)

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        ids = sorted(self._vectors)
        return [
            (vid, self._metadatas.get(vid, {})) for vid in ids[offset : offset + limit]
        ]

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        return [
            (vid, self._metadatas.get(vid, {})) for vid in ids if vid in self._vectors
        ]

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if id in self._vectors:
            self._metadatas[id] = {**self._metadatas.get(id, {}), **metadata}
            self._persist()

    async def clear(self) -> None:
        self._vectors.clear()
        self._metadatas.clear()
        self._rebuild()
        self._persist()
