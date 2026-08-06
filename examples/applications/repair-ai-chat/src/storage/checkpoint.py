"""Session persistence — JSON-file checkpoints for durable conversations.

A checkpoint is saved after every node, so a crash never loses work and a
conversation can continue across process restarts by re-using its session
id.  Swap :func:`build_checkpointer` for
:class:`~draf.checkpoint.SQLiteCheckpointer` when you need a real database.
"""

from __future__ import annotations

from pathlib import Path

from draf.checkpoint import (
    Checkpointer,
    JSONFileCheckpointer,
    PGCheckpointer,
    SQLiteCheckpointer,
)

#: Relative to the example root; points at ``data/checkpoints/``.
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints"


def _is_dsn(value: str | Path | None) -> bool:
    """True when *value* is a Postgres connection string (vs a file path)."""
    return str(value or "").startswith(("postgres://", "postgresql://"))


def build_checkpointer(
    directory: str | Path | None = None,
    *,
    checkpoint_db: str | Path | None = None,
) -> Checkpointer:
    """Build the session checkpointer used by ``graph.run(..., checkpointer=...)``.

    *checkpoint_db* selects the durable backend: a Postgres DSN
    (``postgres://...``) uses :class:`PGCheckpointer`, a path uses
    :class:`SQLiteCheckpointer`, and ``None`` keeps the per-session JSON
    files.
    """
    if _is_dsn(checkpoint_db):
        return PGCheckpointer(str(checkpoint_db))
    if checkpoint_db is not None:
        return SQLiteCheckpointer(str(checkpoint_db))
    return JSONFileCheckpointer(str(directory or DEFAULT_CHECKPOINT_DIR))
