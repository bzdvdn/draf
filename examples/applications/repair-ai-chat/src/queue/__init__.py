"""Background jobs for the repair-ai-chat application.

The Celery app lives in :mod:`src.queue.celery_app`; the re-ingest logic it
runs lives in :mod:`src.queue.ingest`.  Keeping the logic import-free of
Celery means the API process and offline tests never need the broker.
"""
