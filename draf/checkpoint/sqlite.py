"""SQLite checkpointing — stdlib only, shared file format with the RAG store."""

import json
import sqlite3
from pathlib import Path

from draf.checkpoint.base import Checkpoint, Checkpointer


class SQLiteCheckpointer(Checkpointer):
    """Store checkpoints in a SQLite database.

    Uses one row per checkpoint ID, keyed by the ID as primary key.
    Each ``save`` is a single INSERT .. ON CONFLICT REPLACE transaction,
    so a crash leaves either the old or the new row, never a mix.

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                next_node_id TEXT,
                iteration INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    async def save(self, checkpoint_id: str, checkpoint: Checkpoint) -> None:
        self._conn.execute(
            """
            INSERT INTO checkpoints (checkpoint_id, state, next_node_id, iteration)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(checkpoint_id) DO UPDATE SET
                state = excluded.state,
                next_node_id = excluded.next_node_id,
                iteration = excluded.iteration
            """,
            (
                checkpoint_id,
                json.dumps(checkpoint.state, ensure_ascii=False),
                checkpoint.next_node_id,
                checkpoint.iteration,
            ),
        )
        self._conn.commit()

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT state, next_node_id, iteration FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            state=json.loads(row[0]),
            next_node_id=row[1],
            iteration=row[2],
        )

    async def delete(self, checkpoint_id: str) -> None:
        self._conn.execute(
            "DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
        )
        self._conn.commit()
