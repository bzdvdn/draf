"""Typed application settings loaded from the environment / ``.env``.

Override any value with an environment variable prefixed ``DRAF_``
(e.g. ``DRAF_MODEL=llama3.1:8b``) or a line in a local ``.env``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the service-desk router."""

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
    app_title: str = "Service Desk AI"
    app_description: str = (
        "Default-supervisor router over billing / incident / deploy specialists, "
        "each grounded in its own knowledge base."
    )
    version: str = "0.1.0"

    #: Durable session storage path (None = project default ``data/checkpoints``).
    checkpoint_dir: str | None = None

    #: Trace-dashboard persistence (None = ``data/traces.db`` next to the app).
    #: The dashboard UI is mounted by ``app.py`` under ``traces_prefix``.
    traces_db: str | None = None
    traces_prefix: str = "/obs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings (the environment is read once per process)."""
    return Settings()
