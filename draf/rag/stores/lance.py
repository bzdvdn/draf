"""LanceDB vector store — requires ``lancedb``."""

from __future__ import annotations

import json

import numpy as np

from draf.rag.base import VectorStore, finalize_results


def _sql_literal(value: str) -> str:
    """SQL string literal for a LanceDB delete/update WHERE clause."""
    return json.dumps(value)


class LanceVectorStore(VectorStore):
    """Vector store backed by LanceDB (embedded, columnar).

    Requires the ``lancedb`` package. Metadata is stored as a JSON string
    column; filters and hybrid scores are applied after retrieval.
    """

    def __init__(
        self, path: str = "./lance", table: str = "vectors", dim: int | None = None
    ):
        import lancedb

        self._db = lancedb.connect(path)
        self.table = table
        self.dim = dim
        self._tbl = None
        try:
            self._tbl = self._db.open_table(table)
        except Exception:
            self._tbl = None

    def _ensure_dim(self, vec: list[float], vid: str) -> None:
        if self.dim is not None and len(vec) != self.dim:
            msg = f"vector for '{vid}' has dim {len(vec)}, expected {self.dim}"
            raise ValueError(msg)
        if self.dim is None:
            self.dim = len(vec)

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        if not vectors:
            return
        rows = []
        for vid, vec, meta in vectors:
            self._ensure_dim(vec, vid)
            rows.append(
                {
                    "id": vid,
                    "vector": np.asarray(vec, dtype="float32"),
                    "meta": json.dumps(meta, ensure_ascii=False),
                    "text": meta.get("text", ""),
                }
            )
        if self._tbl is None:
            self._tbl = self._db.create_table(self.table, data=rows)
        else:
            self._tbl.add(rows)

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        if self._tbl is None or self._tbl.count_rows() == 0:
            return []
        n_scan = max(k, k * 4) if (filter or hybrid) else k
        res = (
            self._tbl.search(np.asarray(query, dtype="float32"))
            .metric("cosine")
            .limit(n_scan)
            .select(["id", "meta", "text", "_distance"])
            .to_list()
        )
        candidates = []
        for r in res:
            try:
                meta = json.loads(r["meta"])
            except json.JSONDecodeError:
                meta = {}
            candidates.append((r["id"], 1.0 - float(r["_distance"]), meta))
        return finalize_results(candidates, k, filter, hybrid, query_text)

    async def delete(self, ids: list[str]) -> None:
        if not ids or self._tbl is None:
            return
        where = " OR ".join(f"id = {_sql_literal(vid)}" for vid in ids)
        self._tbl.delete(where)

    async def count(self) -> int:
        if self._tbl is None:
            return 0
        return self._tbl.count_rows()

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        if self._tbl is None:
            return []
        df = self._tbl.to_arrow().select(["id", "meta"]).to_pandas().sort_values("id")
        out = []
        for r in df.iloc[offset : offset + limit].itertuples(index=False):
            try:
                meta = json.loads(r.meta)
            except json.JSONDecodeError:
                meta = {}
            out.append((r.id, meta))
        return out

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        if not ids or self._tbl is None:
            return []
        wanted = set(ids)
        df = self._tbl.to_arrow().select(["id", "meta"]).to_pandas()
        out = []
        for r in df.itertuples(index=False):
            if r.id in wanted:
                try:
                    meta = json.loads(r.meta)
                except json.JSONDecodeError:
                    meta = {}
                out.append((r.id, meta))
        return out

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if self._tbl is None:
            return
        df = self._tbl.to_arrow().select(["id", "meta"]).to_pandas()
        row = df[df["id"] == id]
        if row.empty:
            return
        try:
            current = json.loads(row.iloc[0]["meta"])
        except json.JSONDecodeError:
            current = {}
        merged = {**current, **metadata}
        self._tbl.update(
            where=f"id = {_sql_literal(id)}",
            values={
                "meta": json.dumps(merged, ensure_ascii=False),
                "text": merged.get("text", ""),
            },
        )

    async def clear(self) -> None:
        if self._tbl is not None:
            self._db.drop_table(self.table)
            self._tbl = None
