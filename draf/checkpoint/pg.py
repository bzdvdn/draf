"""PostgreSQL checkpointing — requires ``asyncpg`` (``draf[pg-checkpoint]``)."""

import json

from draf.checkpoint.base import Checkpoint, Checkpointer


class PGCheckpointer(Checkpointer):
    """Store checkpoints in a PostgreSQL table.

    Requires ``asyncpg`` (install via ``draf[pg-checkpoint]``). The
    table ``checkpoints`` is created lazily on first use.

    Args:
        dsn: PostgreSQL connection string.
        table: Table name (default ``"checkpoints"``).
    """

    def __init__(self, dsn: str, table: str = "checkpoints"):
        import importlib.util

        if importlib.util.find_spec("asyncpg") is None:
            raise ImportError("install asyncpg for PGCheckpointer")
        self._dsn = dsn
        self._table = table

    async def _connect(self):
        import asyncpg

        conn = await asyncpg.connect(self._dsn)
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                checkpoint_id TEXT PRIMARY KEY,
                state JSONB NOT NULL,
                next_node_id TEXT,
                iteration INTEGER NOT NULL
            )
            """
        )
        return conn

    async def save(self, checkpoint_id: str, checkpoint: Checkpoint) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                f"""
                INSERT INTO {self._table} (checkpoint_id, state, next_node_id, iteration)
                VALUES ($1, $2::jsonb, $3, $4)
                ON CONFLICT(checkpoint_id) DO UPDATE SET
                    state = EXCLUDED.state,
                    next_node_id = EXCLUDED.next_node_id,
                    iteration = EXCLUDED.iteration
                """,
                checkpoint_id,
                json.dumps(checkpoint.state, ensure_ascii=False),
                checkpoint.next_node_id,
                checkpoint.iteration,
            )
        finally:
            await conn.close()

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                f"SELECT state, next_node_id, iteration FROM {self._table} "
                f"WHERE checkpoint_id = $1",
                checkpoint_id,
            )
            if row is None:
                return None
            return Checkpoint(
                state=json.loads(row["state"]),
                next_node_id=row["next_node_id"],
                iteration=row["iteration"],
            )
        finally:
            await conn.close()

    async def delete(self, checkpoint_id: str) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE checkpoint_id = $1", checkpoint_id
            )
        finally:
            await conn.close()
