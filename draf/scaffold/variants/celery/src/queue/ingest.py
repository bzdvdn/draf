"""Change-detection + re-ingest of the document catalog (Celery-free core).

The *beat* scheduler calls :func:`reingest_if_changed` on an interval.  It
re-reads the ``data/documents`` seed files and rebuilds the durable vector
store only when a fingerprint (content hash) of the sources differs from
the last run — an unchanged catalog is a no-op that makes no embedding
calls.

Kept free of Celery imports so tests and the API process can use it without
the task broker installed.  Without the ``rag`` variant there is no catalog
to rebuild: the function reports ``{"status": "no-catalog"}``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from src.config.config import get_settings


def _fingerprint(paths: Sequence[Path | str]) -> str:
    """Stable content hash of the given source files."""
    digest = hashlib.sha256()
    for path in paths:
        p = Path(path)
        if p.exists():
            digest.update(p.read_bytes())
    return digest.hexdigest()


def _sources() -> list[Path]:
    """The seed files under ``data/documents`` that drive re-ingestion."""
    from src.rag.wiring import DEFAULT_DOCUMENTS

    if not DEFAULT_DOCUMENTS.is_dir():
        return []
    return sorted(p for p in DEFAULT_DOCUMENTS.iterdir() if p.is_file())


def _state_path() -> Path:
    """Where the last-run fingerprint is stored (alongside the schedule DB)."""
    settings = get_settings()
    if settings.beat_schedule:
        return Path(settings.beat_schedule).with_name("ingest_state.json")
    return Path(__file__).resolve().parents[2] / "data" / "ingest_state.json"


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


async def reingest_if_changed(*, force: bool = False, batch_size: int = 250) -> dict:
    """Rebuild the catalog store when the seed documents changed since last run.

    Returns a status dict suitable for a Celery result:

    * ``{"status": "ok", "queued", "added", "batches", "stored"}`` — sources
      changed (or *force*), store was rebuilt.
    * ``{"status": "unchanged", "stored"}`` — no content change, no work.
    * ``{"status": "no-catalog", "stored": 0}`` — the ``rag`` variant is not
      installed, so there is nothing to re-embed.
    """
    try:
        from src.rag.wiring import build_catalog
    except ImportError:
        return {"status": "no-catalog", "stored": 0}
    catalog = build_catalog()
    sources = _sources()
    if not sources:
        return {"status": "no-catalog", "stored": 0}
    state = _load_state(_state_path())
    if not force and state.get("fingerprint") == _fingerprint(sources):
        return {"status": "unchanged", "stored": catalog.stored}
    report = await catalog.rebuild(batch_size=batch_size)
    _save_state(_state_path(), {"fingerprint": _fingerprint(sources)})
    return {
        "status": "ok",
        "queued": report.queued,
        "added": report.added,
        "batches": report.batches,
        "stored": report.stored,
    }
