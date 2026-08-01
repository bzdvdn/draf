"""Typed application settings loaded from the environment / ``.env``.

The daemon needs the LLM ``provider``/``model``, the durable session storage
path, and where to look for pending jobs and write results.  Override any
value with an environment variable prefixed ``DRAF_`` or a line in a local
``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the {{PROJECT_NAME}} daemon."""

    model_config = SettingsConfigDict(
        env_prefix="DRAF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: LLM provider (``"ollama"``, ``"openai"``, ...) and default model.
    provider: str = "ollama"
    model: str = "llama3.1:8b"

    #: Durable session storage path (None = project default ``data/checkpoints``).
    checkpoint_dir: str | None = None

    #: Seconds to sleep between queue polls.
    poll_interval: float = 2.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings (the environment is read once per process)."""
    return Settings()
