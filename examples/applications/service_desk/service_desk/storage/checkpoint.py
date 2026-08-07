"""Session persistence — JSON-file checkpoints for durable conversations.

A checkpoint is saved after every node, so a crash never loses work and a
conversation can continue across process restarts by re-using its session
id.  Swap :func:`build_checkpointer` for
:class:`~teff.checkpoint.SQLiteCheckpointer` when you need a real database.
"""

from __future__ import annotations

from pathlib import Path

from teff.checkpoint import JSONFileCheckpointer

#: Relative to the project root; points at ``data/checkpoints/``.
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints"


def build_checkpointer(directory: str | Path | None = None) -> JSONFileCheckpointer:
    """Build the session checkpointer used by ``graph.run(..., checkpointer=...)``."""
    return JSONFileCheckpointer(str(directory or DEFAULT_CHECKPOINT_DIR))
