"""Typed application settings loaded from the environment / ``.env``.

Override any value with an environment variable prefixed ``TEFF_``
(e.g. ``TEFF_MODEL=llama3.1:8b``) or a line in a local ``.env``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the simple router."""

    model_config = SettingsConfigDict(
        env_prefix="TEFF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: LLM provider (``"ollama"``, ``"openai"``, ...) and default model.
    provider: str = "ollama"
    model: str = "llama3.1:8b"

    #: Durable session storage path (None = project default ``data/checkpoints``).
    checkpoint_dir: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings (the environment is read once per process)."""
    return Settings()
