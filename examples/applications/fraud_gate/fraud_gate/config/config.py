"""Typed application settings from the environment / ``.env``.

Override any value with an environment variable prefixed ``DRAF_``
(e.g. ``DRAF_PORT=8080``) or a line in a local ``.env``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the fraud-gate server."""

    model_config = SettingsConfigDict(
        env_prefix="DRAF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: LLM provider (``"ollama"``, ``"openai"``, ...) and default model.
    provider: str = "ollama"
    model: str = "llama3.1:8b"

    #: Decision thresholds used by the router node (overridable for tests).
    review_threshold: float = 0.6
    deny_threshold: float = 0.9

    #: When set, the review routers require the ``X-API-Key`` header.
    api_key: str | None = None

    #: Bind address for ``main.py``.
    host: str = "127.0.0.1"
    port: int = 8001

    #: App metadata surfaced by FastAPI.
    app_title: str = "Fraud Gate AI"
    app_description: str = (
        "Production-style payment screening — a Command-routing gate that "
        "auto-approves, sends mid-risk payments to a human analyst or blocks "
        "fraud up front."
    )
    version: str = "0.1.0"

    #: Durable session storage path (None = project default ``data/checkpoints``).
    checkpoint_dir: str | None = None

    #: Trace DB for the observability dashboard (None = ``data/traces.db``).
    traces_db: str | None = None
    #: Mount point for the trace dashboard UI/APIs.
    traces_prefix: str = "/obs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings (the environment is read once per process)."""
    return Settings()
