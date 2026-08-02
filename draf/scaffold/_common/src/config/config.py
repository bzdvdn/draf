"""Typed application settings loaded from the environment / ``.env``.

One superset shared by every app kind (fastapi / cli / daemon): each entry
point reads only the fields it needs.  Override any value with an
environment variable prefixed ``DRAF_`` (e.g. ``DRAF_PORT=9000``) or a line
in a local ``.env`` file.

HOW TO EXTEND
    Add a knob here when a new module needs configuration (a provider URL, a
    feature flag, a storage path).  Keep the names discoverable — every field
    is documented with a ``#:`` comment so ``Settings()`` stays self-explanatory.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the {{PROJECT_NAME}} app."""

    model_config = SettingsConfigDict(
        env_prefix="DRAF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: LLM provider (``"ollama"``, ``"openai"``, ...) and default model.
    provider: str = "ollama"
    model: str = "llama3.1:8b"

    #: When set, the chat/run routers require the ``X-API-Key`` header.
    api_key: str | None = None

    #: Bind address for ``main.py``.
    host: str = "127.0.0.1"
    port: int = 8000

    #: App metadata surfaced by FastAPI.
    app_title: str = "{{PROJECT_NAME}}"
    app_description: str = "A production draf supervisor app."
    version: str = "0.1.0"

    #: Durable session storage path (None = project default ``data/checkpoints``).
    checkpoint_dir: str | None = None

    #: SQLite persistence (shared file so API + workers read the same data).
    #: When set, ``checkpoint_db`` replaces the JSON-file checkpointer.
    checkpoint_db: str | None = None

    #: PostgreSQL DSN (``postgres://...``).  When set, it wins over the SQLite
    #: paths: sessions and RAG vectors live in Postgres, shared by every process.
    database_url: str | None = None

    #: RAG embedder provider and top-k used by the catalog (``rag`` variant).
    rag_embedder: str = "ollama"
    rag_top_k: int = 3

    #: Seconds to sleep between queue polls (daemon kind).
    poll_interval: float = 2.0

    #: Redis broker URL for the Celery worker/beat (``celery`` variant).
    #: ``None`` disables background jobs.
    redis_url: str | None = None

    #: Where Celery beat keeps its schedule DB (writable dir).
    beat_schedule: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings (the environment is read once per process)."""
    return Settings()
