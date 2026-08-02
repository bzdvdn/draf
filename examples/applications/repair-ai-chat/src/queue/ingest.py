"""Change-detection + re-ingest of the material catalog (Celery-free core).

The *beat* scheduler calls :func:`reingest_if_changed` on an interval.  It
re-reads the seed CSVs and rebuilds the durable vector store only when a
fingerprint (content hash) of the sources differs from the last run — an
unchanged catalog is a no-op that makes no embedding calls.

Kept free of Celery imports so tests and the API process can use it
without the task broker installed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from src.config.config import get_settings
from src.core.deps import (
    DEFAULT_CATALOG,
    DEFAULT_CATALOG_DB,
    DEFAULT_PRICE_LIST,
    build_deps,
)
from src.rag.catalog import MaterialCatalog


def _fingerprint(paths: Sequence[Path | str]) -> str:
    """Stable content hash of the given source files."""
    digest = hashlib.sha256()
    for path in paths:
        p = Path(path)
        if p.exists():
            digest.update(p.read_bytes())
    return digest.hexdigest()


def _state_path() -> Path:
    """Where the last-run fingerprint is stored (alongside the durable DB)."""
    settings = get_settings()
    if settings.catalog_ingest_state:
        return Path(settings.catalog_ingest_state)
    db = settings.database_url or settings.catalog_db or DEFAULT_CATALOG_DB
    if str(db).startswith(("postgres://", "postgresql://")):
        return Path(__file__).resolve().parents[2] / "data" / "ingest_state.json"
    return Path(db).with_suffix(".ingest_state.json")


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


async def reingest_if_changed(
    *,
    force: bool = False,
    batch_size: int = 250,
    catalog: MaterialCatalog | None = None,
) -> dict:
    """Rebuild the catalog store when the seed CSVs changed since last run.

    Returns a status dict suitable for a Celery result:

    * ``{"status": "ok", "queued", "added", "batches", "stored"}`` — sources
      changed (or *force*), store was rebuilt.
    * ``{"status": "unchanged", "stored"}`` — no content change, no work.

    *catalog* is an optional pre-built :class:`MaterialCatalog` (used by
    tests to inject a stubbed embedder/store); when omitted it is built
    from the configured durable store like the API does.
    """
    settings = get_settings()
    if catalog is None:
        _services, catalog = build_deps(
            catalog_csv=settings.catalog_csv,
            provider=settings.rag_embedder or settings.provider,
            catalog_db=settings.database_url or settings.catalog_db,
        )
    sources = [DEFAULT_CATALOG, DEFAULT_PRICE_LIST]
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
