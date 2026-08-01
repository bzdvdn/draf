"""SQLite checkpointing — stdlib only, shared file format with the RAG store."""

import json
import sqlite3
from pathlib import Path

from draf.checkpoint.base import DEFAULT_OWNER, Checkpoint, Checkpointer


class SQLiteCheckpointer(Checkpointer):
    """Store checkpoints in a SQLite database.

    Uses one row per ``(owner, checkpoint_id)`` pair — a composite primary
    key, so the same ID can belong to different owners (users/tenants)
    without colliding.  Each ``save`` is a single ``INSERT .. ON CONFLICT
    REPLACE`` transaction, so a crash leaves either the old or the new row,
    never a mix.  Existing single-owner databases are migrated in place:
    their rows move under :data:`~draf.checkpoint.DEFAULT_OWNER`.

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._migrate()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                owner TEXT NOT NULL DEFAULT 'default',
                checkpoint_id TEXT NOT NULL,
                state TEXT NOT NULL,
                next_node_id TEXT,
                iteration INTEGER NOT NULL,
                PRIMARY KEY (owner, checkpoint_id)
            )
            """
        )
        self._conn.commit()

    def _migrate(self) -> None:
        """Migrate a legacy single-owner table to the owner-scoped schema."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        ).fetchone()
        if row is None:
            return
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(checkpoints)")]
        if "owner" in cols:
            return
        self._conn.execute("ALTER TABLE checkpoints RENAME TO checkpoints_legacy")
        self._conn.execute(
            """
            CREATE TABLE checkpoints (
                owner TEXT NOT NULL DEFAULT 'default',
                checkpoint_id TEXT NOT NULL,
                state TEXT NOT NULL,
                next_node_id TEXT,
                iteration INTEGER NOT NULL,
                PRIMARY KEY (owner, checkpoint_id)
            )
            """
        )
        self._conn.execute(
            """
            INSERT INTO checkpoints (owner, checkpoint_id, state, next_node_id, iteration)
            SELECT 'default', checkpoint_id, state, next_node_id, iteration
            FROM checkpoints_legacy
            """
        )
        self._conn.execute("DROP TABLE checkpoints_legacy")
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    async def save(
        self,
        checkpoint_id: str,
        checkpoint: Checkpoint,
        *,
        owner: str = DEFAULT_OWNER,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO checkpoints (owner, checkpoint_id, state, next_node_id, iteration)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(owner, checkpoint_id) DO UPDATE SET
                state = excluded.state,
                next_node_id = excluded.next_node_id,
                iteration = excluded.iteration
            """,
            (
                owner,
                checkpoint_id,
                json.dumps(checkpoint.state, ensure_ascii=False),
                checkpoint.next_node_id,
                checkpoint.iteration,
            ),
        )
        self._conn.commit()

    async def load(
        self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER
    ) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT state, next_node_id, iteration FROM checkpoints "
            "WHERE owner = ? AND checkpoint_id = ?",
            (owner, checkpoint_id),
        ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            state=json.loads(row[0]),
            next_node_id=row[1],
            iteration=row[2],
        )

    async def delete(self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER) -> None:
        self._conn.execute(
            "DELETE FROM checkpoints WHERE owner = ? AND checkpoint_id = ?",
            (owner, checkpoint_id),
        )
        self._conn.commit()

    async def list(self, owner: str = DEFAULT_OWNER) -> list[str]:
        """Return all checkpoint IDs persisted for *owner*."""
        rows = self._conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE owner = ? ORDER BY checkpoint_id",
            (owner,),
        ).fetchall()
        return [r[0] for r in rows]
