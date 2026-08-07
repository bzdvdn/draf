"""Session persistence — durable checkpoints for resumable reviews."""

from __future__ import annotations

from pathlib import Path

from teff.checkpoint import JSONFileCheckpointer

#: Relative to the app root; points at ``data/checkpoints/``.
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints"


def build_checkpointer(directory: str | Path | None = None) -> JSONFileCheckpointer:
    """Build the session checkpointer used by ``graph.run(checkpointer=...)``."""
    return JSONFileCheckpointer(str(directory or DEFAULT_CHECKPOINT_DIR))
