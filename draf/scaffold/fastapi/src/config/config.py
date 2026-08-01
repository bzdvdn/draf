"""Typed application settings loaded from the environment / ``.env``.

Every knob the server needs lives here — the LLM ``provider``/``model``,
the optional ``api_key``, the bind address, app metadata and the storage
and RAG paths — so ``main.py``, ``app.py`` and ``cli.py`` share a single
source of truth.  Override any value with an environment variable prefixed
``DRAF_`` (e.g. ``DRAF_PORT=9000``) or a line in a local ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the {{PROJECT_NAME}} service."""

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings (the environment is read once per process)."""
    return Settings()
