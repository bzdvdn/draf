"""JSON-file checkpointing — zero dependencies, atomic via tempfile + rename."""

import json
import os
from pathlib import Path

from draf.checkpoint.base import (
    Checkpoint,
    Checkpointer,
    checkpoint_from_dict,
    checkpoint_to_dict,
)


class JSONFileCheckpointer(Checkpointer):
    """Store checkpoints as one JSON file per checkpoint ID.

    Writes go to a temp file in the same directory and are atomically
    renamed over the target, so a crash never leaves a corrupt file.

    Args:
        directory: Directory to store checkpoint files in (created if needed).
        suffix: File suffix for checkpoint files (default ``".json"``).
    """

    def __init__(self, directory: str, suffix: str = ".json"):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._suffix = suffix

    def _path(self, checkpoint_id: str) -> Path:
        safe = checkpoint_id.replace(os.sep, "_").replace("/", "_")
        return self._directory / f"{safe}{self._suffix}"

    async def save(self, checkpoint_id: str, checkpoint: Checkpoint) -> None:
        target = self._path(checkpoint_id)
        tmp = target.with_suffix(f"{self._suffix}.tmp")
        tmp.write_text(
            json.dumps(checkpoint_to_dict(checkpoint), ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, target)

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        path = self._path(checkpoint_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return checkpoint_from_dict(data)

    async def delete(self, checkpoint_id: str) -> None:
        path = self._path(checkpoint_id)
        if path.exists():
            path.unlink()
