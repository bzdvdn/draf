"""Celery application for the {{PROJECT_NAME}} background jobs.

Run the worker and the beat scheduler (from compose or your shell)::

    celery -A src.queue.celery_app:celery_app worker --loglevel=info
    celery -A src.queue.celery_app:celery_app beat   --loglevel=info

``beat`` triggers :func:`src.queue.ingest.reingest_if_changed` on an
interval so the RAG catalog picks up changed documents automatically.
Requires the ``queue`` extra (``uv sync --extra queue`` / ``draf[queue]``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from celery import Celery
from src.config.config import get_settings
from src.queue.ingest import reingest_if_changed

_settings = get_settings()

celery_app = Celery(
    "{{project_slug}}",
    broker=_settings.redis_url or "redis://localhost:6379/0",
    backend=_settings.redis_url or "redis://localhost:6379/0",
)
celery_app.conf.timezone = "UTC"
celery_app.conf.beat_schedule = {
    "reingest-documents-every-5-min": {
        "task": "{{project_slug}}.reingest_documents",
        "schedule": 300.0,
    },
}
# Beat persists its schedule DB to disk; default to the project ``data/``
# dir, which compose overrides to the writable ``/data`` volume.
if _settings.beat_schedule:
    celery_app.conf.beat_schedule_filename = str(_settings.beat_schedule)
else:
    celery_app.conf.beat_schedule_filename = str(
        Path(__file__).resolve().parents[2] / "data" / "celerybeat-schedule"
    )
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.autodiscover_tasks([])


@celery_app.task(name="{{project_slug}}.reingest_documents")
def reingest_documents(force: bool = False) -> dict:
    """Re-embed the document catalog when the seed files changed on disk."""
    return asyncio.run(reingest_if_changed(force=force))
