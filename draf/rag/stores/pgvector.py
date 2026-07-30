"""PostgreSQL + pgvector store — requires ``asyncpg`` + ``pgvector``."""

from draf.rag.base import VectorStore


class PGVectorStore(VectorStore):
    """Vector store backed by PostgreSQL with pgvector extension.

    Requires ``asyncpg`` and ``sqlalchemy`` (install via ``draf[embedding]``).
    """

    def __init__(self, dsn: str, table: str = "draf_vectors"):
        import importlib.util
        if importlib.util.find_spec("asyncpg") is None:
            raise ImportError("install asyncpg + sqlalchemy + pgvector for PGVectorStore")
        self._dsn = dsn
        self._table = table

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        import asyncpg
        conn = await asyncpg.connect(self._dsn)
        try:
            for vid, vec, meta in vectors:
                await conn.execute(
                    f"INSERT INTO {self._table} (doc_id, embedding, metadata) VALUES ($1, $2, $3)",
                    vid, vec, meta,
                )
        finally:
            await conn.close()

    async def search(self, query: list[float], k: int = 10) -> list[tuple[str, float, dict]]:
        import asyncpg
        conn = await asyncpg.connect(self._dsn)
        try:
            rows = await conn.fetch(
                f"SELECT doc_id, 1 - (embedding <=> $1::vector) AS score, metadata "
                f"FROM {self._table} ORDER BY embedding <=> $1::vector LIMIT $2",
                query, k,
            )
            return [(r["doc_id"], float(r["score"]), dict(r["metadata"])) for r in rows]
        finally:
            await conn.close()

    async def delete(self, ids: list[str]) -> None:
        import asyncpg
        conn = await asyncpg.connect(self._dsn)
        try:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE doc_id = ANY($1)", ids,
            )
        finally:
            await conn.close()
