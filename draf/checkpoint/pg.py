"""PostgreSQL checkpointing — requires ``asyncpg`` (``draf[pg-checkpoint]``)."""

import json

from draf.checkpoint.base import DEFAULT_OWNER, Checkpoint, Checkpointer


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
                owner TEXT NOT NULL DEFAULT 'default',
                checkpoint_id TEXT NOT NULL,
                state JSONB NOT NULL,
                next_node_id TEXT,
                iteration INTEGER NOT NULL,
                updated_at DOUBLE PRECISION,
                PRIMARY KEY (owner, checkpoint_id)
            )
            """
        )
        return conn

    async def save(
        self,
        checkpoint_id: str,
        checkpoint: Checkpoint,
        *,
        owner: str = DEFAULT_OWNER,
    ) -> None:
        import time

        conn = await self._connect()
        try:
            await conn.execute(
                f"""
                INSERT INTO {self._table} (owner, checkpoint_id, state, next_node_id, iteration, updated_at)
                VALUES ($1, $2, $3::jsonb, $4, $5, $6)
                ON CONFLICT(owner, checkpoint_id) DO UPDATE SET
                    state = EXCLUDED.state,
                    next_node_id = EXCLUDED.next_node_id,
                    iteration = EXCLUDED.iteration,
                    updated_at = EXCLUDED.updated_at
                """,
                owner,
                checkpoint_id,
                json.dumps(checkpoint.state, ensure_ascii=False),
                checkpoint.next_node_id,
                checkpoint.iteration,
                time.time(),
            )
        finally:
            await conn.close()

    async def load(
        self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER
    ) -> Checkpoint | None:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                f"SELECT state, next_node_id, iteration FROM {self._table} "
                f"WHERE owner = $1 AND checkpoint_id = $2",
                owner,
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

    async def delete(self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE owner = $1 AND checkpoint_id = $2",
                owner,
                checkpoint_id,
            )
        finally:
            await conn.close()

    async def list(self, owner: str = DEFAULT_OWNER) -> list[str]:
        """Return all checkpoint IDs persisted for *owner*."""
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                f"SELECT checkpoint_id FROM {self._table} "
                f"WHERE owner = $1 ORDER BY checkpoint_id",
                owner,
            )
            return [r["checkpoint_id"] for r in rows]
        finally:
            await conn.close()

    async def cleanup(
        self,
        *,
        owner: str | None = None,
        max_age: float | None = None,
        keep_last: int | None = None,
    ) -> int:
        """Delete stale checkpoints; returns how many were removed."""
        import time

        if max_age is None and keep_last is None:
            return 0
        removed = 0
        now = time.time()
        conn = await self._connect()
        try:
            if owner is not None:
                owners = [owner]
            else:
                rows = await conn.fetch(f"SELECT DISTINCT owner FROM {self._table}")
                owners = [r["owner"] for r in rows]
            for own in owners:
                if max_age is not None:
                    result = await conn.execute(
                        f"DELETE FROM {self._table} WHERE owner = $1 AND "
                        f"COALESCE(updated_at, 0) < $2",
                        own,
                        now - max_age,
                    )
                    removed += int(result.split(" ", 1)[0] or 0)
                if keep_last is not None:
                    stale = await conn.fetch(
                        f"SELECT checkpoint_id FROM {self._table} WHERE owner = $1 "
                        f"ORDER BY COALESCE(updated_at, 0) DESC OFFSET $2",
                        own,
                        keep_last,
                    )
                    for row in stale:
                        await conn.execute(
                            f"DELETE FROM {self._table} "
                            f"WHERE owner = $1 AND checkpoint_id = $2",
                            own,
                            row["checkpoint_id"],
                        )
                        removed += 1
        finally:
            await conn.close()
        return removed
