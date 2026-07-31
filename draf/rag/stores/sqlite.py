"""SQLite vector store — file persistence with zero extra dependencies."""

import json
import sqlite3

from draf.rag.base import VectorStore, cosine_similarity


class SQLiteVectorStore(VectorStore):
    """File-persistent vector store backed by SQLite (stdlib only).

    Vectors are stored as JSON blobs in a local ``.db`` file. Search is a
    brute-force cosine similarity scan over all rows — suitable for small
    to medium collections where you want persistence without installing a
    heavy vector database.

    Args:
        path: Path to the SQLite database file.
        dim: Vector dimensionality (used as a sanity check on add).
    """

    def __init__(self, path: str = "./vectors.db", dim: int | None = None):
        self.path = path
        self.dim = dim
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "id TEXT PRIMARY KEY, vector TEXT NOT NULL, metadata TEXT NOT NULL)"
        )

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        rows = []
        for vid, vec, meta in vectors:
            if self.dim is not None and len(vec) != self.dim:
                msg = f"vector for '{vid}' has dim {len(vec)}, expected {self.dim}"
                raise ValueError(msg)
            rows.append((vid, json.dumps(vec), json.dumps(meta, ensure_ascii=False)))
        self._conn.executemany(
            "INSERT OR REPLACE INTO vectors (id, vector, metadata) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    async def search(
        self, query: list[float], k: int = 10
    ) -> list[tuple[str, float, dict]]:
        rows = self._conn.execute("SELECT id, vector, metadata FROM vectors").fetchall()
        scored = []
        for vid, vec_json, meta_json in rows:
            vec = json.loads(vec_json)
            sim = cosine_similarity(query, vec)
            scored.append((vid, sim, json.loads(meta_json)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    async def delete(self, ids: list[str]) -> None:
        self._conn.executemany("DELETE FROM vectors WHERE id = ?", [(i,) for i in ids])
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
