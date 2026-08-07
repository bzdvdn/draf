"""Filesystem/env tools — list, glob, env vars, and time."""

import datetime
import glob
import os
import zoneinfo

from teff.tool.tool import Tool

_SECRET_HINTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "dsn",
    "database_url",
    "connection_string",
)


class ListDirTool(Tool):
    """List the contents of a directory."""

    name = "list_dir"
    description = "List files and directories in a path"

    def run(self, path: str = ".", recursive: bool = False) -> str:  # type: ignore[override]
        if not os.path.isdir(path):
            raise ValueError(f"not a directory: {path}")
        entries: list[str] = []
        if recursive:
            for root, dirs, files in os.walk(path):
                for name in sorted(dirs + files):
                    entries.append(os.path.join(root, name))
        else:
            entries = sorted(os.listdir(path))
        return "\n".join(entries) if entries else "empty directory"


class GlobTool(Tool):
    """Find files matching a glob pattern."""

    name = "glob"
    description = "Find files matching a glob pattern"

    def run(self, pattern: str = "") -> str:  # type: ignore[override]
        if not pattern:
            raise ValueError("pattern is required")
        matches = sorted(glob.glob(pattern, recursive=True))
        return "\n".join(matches) if matches else "no matches"


class GetEnvTool(Tool):
    """Read an environment variable, masking secret-looking values.

    Values whose names hint at credentials (``TOKEN``, ``API_KEY``,
    ``PASSWORD``, ``DSN``, …) are returned as ``***`` unless the tool is
    configured with ``mask_secrets=False``.
    """

    name = "getenv"
    description = "Read an environment variable (secrets are masked)"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.mask_secrets = cfg.get("mask_secrets", True)

    def run(self, name: str = "") -> str:  # type: ignore[override]
        if not name:
            raise ValueError("name is required")
        value = os.environ.get(name)
        if value is None:
            return "not set"
        if self.mask_secrets and self._is_secret(name):
            return "***"
        return value

    @staticmethod
    def _is_secret(name: str) -> bool:
        lowered = name.lower()
        return any(hint in lowered for hint in _SECRET_HINTS)


class CurrentTimeTool(Tool):
    """Get the current date and time.

    Args:
        config: Optional dict with ``timezone`` — an IANA name such as
            ``"Europe/Moscow"`` (default ``"local"``).
    """

    name = "current_time"
    description = "Get the current date and time"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.timezone = cfg.get("timezone", "local")

    def run(self, timezone: str = "") -> str:  # type: ignore[override]
        tz = timezone or self.timezone
        now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        if tz != "local":
            try:
                now = now.astimezone(zoneinfo.ZoneInfo(tz))
            except zoneinfo.ZoneInfoNotFoundError as e:
                raise ValueError(f"unknown timezone: {tz}") from e
        return now.isoformat(timespec="seconds")
