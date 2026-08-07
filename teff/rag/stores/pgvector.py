"""PostgreSQL + pgvector store — requires ``asyncpg`` + ``pgvector``."""

from __future__ import annotations

from teff.rag.base import VectorStore


class PGVectorStore(VectorStore):
    """Vector store backed by PostgreSQL with pgvector extension.

    Requires ``asyncpg`` + ``pgvector`` (install via ``teff[embedding]``).
    Metadata is stored in a ``jsonb`` column; filters are translated into
    SQL ``WHERE`` clauses on the top-level JSON keys.

    The schema (``vector`` extension + table with a primary key on
    ``doc_id``) is bootstrapped lazily on the first connection, and ``add``
    is idempotent (``ON CONFLICT DO UPDATE``) so re-embedding a document or
    a fresh process re-indexing the same queue never duplicates rows.
    """

    def __init__(self, dsn: str, table: str = "teff_vectors", dim: int = 768):
        import importlib.util

        if importlib.util.find_spec("asyncpg") is None:
            raise ImportError("install asyncpg + pgvector for PGVectorStore")
        self._dsn = dsn
        self._table = table
        self._dim = dim

    async def _connect(self):
        import asyncpg
        import pgvector.asyncpg

        conn = await asyncpg.connect(self._dsn)
        await pgvector.asyncpg.register_vector(conn)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            f"doc_id TEXT PRIMARY KEY, "
            f"embedding vector({self._dim}) NOT NULL, "
            f"metadata JSONB NOT NULL)"
        )
        return conn

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        import json

        conn = await self._connect()
        try:
            await conn.executemany(
                f"INSERT INTO {self._table} (doc_id, embedding, metadata) "
                f"VALUES ($1, $2::vector, $3::jsonb) "
                f"ON CONFLICT (doc_id) DO UPDATE SET "
                f"embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata",
                [
                    (vid, vec, json.dumps(meta, ensure_ascii=False))
                    for vid, vec, meta in vectors
                ],
            )
        finally:
            await conn.close()

    def _where_clause(self, filter: dict | None) -> tuple[str, list]:
        """Build ``(sql_where, params)`` from the filter DSL.

        ``$1`` is reserved for the query vector, so metadata parameters
        start at ``$2``.  Metadata values are compared as text.
        """
        if not filter:
            return "", []
        params: list = []

        def build(node: dict) -> str:
            subs: list[str] = []
            for key, cond in node.items():
                if key in ("$and", "$or"):
                    op = " AND " if key == "$and" else " OR "
                    inner = [build(c) for c in cond]
                    inner = [s for s in inner if s]
                    if inner:
                        subs.append("(" + op.join(inner) + ")")
                elif isinstance(cond, list):
                    ph = ", ".join(
                        "$" + str(len(params) + 2 + i) for i in range(len(cond))
                    )
                    subs.append(f"metadata->>'{key}' IN ({ph})")
                    params.extend(str(c) for c in cond)
                else:
                    params.append(str(cond))
                    subs.append(f"metadata->>'{key}' = ${len(params) + 1}")
            return " AND ".join(subs)

        return build(filter), params

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        import json

        where, params = self._where_clause(filter)
        conn = await self._connect()
        try:
            sql = (
                f"SELECT doc_id, 1 - (embedding <=> $1::vector) AS score, metadata "
                f"FROM {self._table}"
            )
            if where:
                sql += " WHERE " + where
            sql += f" ORDER BY embedding <=> $1::vector LIMIT ${len(params) + 1}"
            rows = await conn.fetch(sql, query, *params, k)
            return [
                (r["doc_id"], float(r["score"]), json.loads(r["metadata"]))
                for r in rows
            ]
        finally:
            await conn.close()

    async def delete(self, ids: list[str]) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE doc_id = ANY($1)",
                ids,
            )
        finally:
            await conn.close()

    async def count(self) -> int:
        conn = await self._connect()
        try:
            row = await conn.fetchval(f"SELECT COUNT(*) FROM {self._table}")
            return int(row)
        finally:
            await conn.close()

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        import json

        conn = await self._connect()
        try:
            rows = await conn.fetch(
                f"SELECT doc_id, metadata FROM {self._table} "
                f"ORDER BY doc_id LIMIT $1 OFFSET $2",
                limit,
                offset,
            )
            return [(r["doc_id"], json.loads(r["metadata"])) for r in rows]
        finally:
            await conn.close()

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        import json

        conn = await self._connect()
        try:
            rows = await conn.fetch(
                f"SELECT doc_id, metadata FROM {self._table} WHERE doc_id = ANY($1)",
                ids,
            )
            return [(r["doc_id"], json.loads(r["metadata"])) for r in rows]
        finally:
            await conn.close()

    async def update_metadata(self, id: str, metadata: dict) -> None:
        import json

        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                f"SELECT metadata FROM {self._table} WHERE doc_id = $1", id
            )
            if row is None:
                return
            merged = {**json.loads(row["metadata"]), **metadata}
            await conn.execute(
                f"UPDATE {self._table} SET metadata = $2::jsonb WHERE doc_id = $1",
                id,
                json.dumps(merged),
            )
        finally:
            await conn.close()

    async def clear(self) -> None:
        conn = await self._connect()
        try:
            await conn.execute(f"DELETE FROM {self._table}")
        finally:
            await conn.close()
