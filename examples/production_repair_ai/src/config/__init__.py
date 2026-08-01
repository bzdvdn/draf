"""Application configuration — environment-driven settings.

Re-exported so the rest of the app only needs ``from src.config import ...``.
"""

from src.config.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
