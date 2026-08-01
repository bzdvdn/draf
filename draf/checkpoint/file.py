"""JSON-file checkpointing — zero dependencies, atomic via tempfile + rename."""

import json
import os
from pathlib import Path

from draf.checkpoint.base import (
    DEFAULT_OWNER,
    Checkpoint,
    Checkpointer,
    checkpoint_from_dict,
    checkpoint_to_dict,
)


class JSONFileCheckpointer(Checkpointer):
    """Store checkpoints as one JSON file per (owner, checkpoint ID).

    Writes go to a temp file in the same directory and are atomically
    renamed over the target, so a crash never leaves a corrupt file.
    Each *owner* gets its own subdirectory, so IDs only need to be
    unique within an owner.  See :class:`~draf.checkpoint.Checkpointer`
    for how to pick an owner.
    """

    def __init__(self, directory: str, suffix: str = ".json"):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._suffix = suffix

    def _path(self, checkpoint_id: str, owner: str = DEFAULT_OWNER) -> Path:
        safe = checkpoint_id.replace(os.sep, "_").replace("/", "_")
        owner_dir = self._directory / self._safe_owner(owner)
        owner_dir.mkdir(parents=True, exist_ok=True)
        return owner_dir / f"{safe}{self._suffix}"

    @staticmethod
    def _safe_owner(owner: str) -> str:
        return owner.replace(os.sep, "_").replace("/", "_").replace(".", "_")

    async def save(
        self,
        checkpoint_id: str,
        checkpoint: Checkpoint,
        *,
        owner: str = DEFAULT_OWNER,
    ) -> None:
        target = self._path(checkpoint_id, owner)
        tmp = target.with_suffix(f"{self._suffix}.tmp")
        tmp.write_text(
            json.dumps(checkpoint_to_dict(checkpoint), ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, target)

    async def load(
        self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER
    ) -> Checkpoint | None:
        path = self._path(checkpoint_id, owner)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return checkpoint_from_dict(data)

    async def delete(self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER) -> None:
        path = self._path(checkpoint_id, owner)
        if path.exists():
            path.unlink()

    async def list(self, owner: str = DEFAULT_OWNER) -> list[str]:
        """Return all checkpoint IDs persisted for *owner*."""
        base = self._directory / self._safe_owner(owner)
        if not base.exists():
            return []
        return sorted(
            p.name[: -len(self._suffix)]
            for p in base.glob(f"*{self._suffix}")
            if not p.name.endswith(f"{self._suffix}.tmp")
        )
