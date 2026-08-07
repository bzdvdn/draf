"""Build a :class:`Checkpointer` from a declarative ``{type, ...}`` config.

Shared by the CLI and a workflow's ``checkpoint:`` block so the two never
drift.  ``path``/``dsn`` values are taken verbatim; relative ``path``
entries should be resolved against the workflow directory before calling
this (see :func:`teff.yaml.checkpointer_from_workflow`).
"""

from __future__ import annotations

import os

from teff.checkpoint import (
    Checkpointer,
    JSONFileCheckpointer,
    PGCheckpointer,
    PGHistoryCheckpointer,
    SQLiteCheckpointer,
    SQLiteHistoryCheckpointer,
)
from teff.errors import ConfigError


def resolve_checkpoint_config(config: dict | None, base_dir: str) -> dict:
    """Return a copy of a ``checkpoint:`` config with relative paths absolved.

    Makes ``path`` keys relative to *base_dir* (the workflow directory) so
    SQLite/JSON checkpoints land next to the workflow that declares them.
    """
    out = dict(config or {})
    path = out.get("path")
    if isinstance(path, str) and not os.path.isabs(path):
        out["path"] = os.path.join(base_dir, path)
    return out


def checkpointer_from_config(config) -> Checkpointer:
    """Instantiate a checkpointer from a ``{type, ...}`` mapping.

    Raises :class:`~teff.errors.ConfigError` for an unknown ``type`` or a
    missing required field (e.g. ``dsn`` for ``pg``).
    """
    if not isinstance(config, dict):
        raise ConfigError("checkpoint must be a mapping with `type`")
    ctype = config.get("type")
    if ctype == "file":
        return JSONFileCheckpointer(config.get("path", "checkpoints"))
    if ctype == "sqlite":
        return SQLiteCheckpointer(config.get("path", "checkpoints.db"))
    if ctype == "sqlite_history":
        return SQLiteHistoryCheckpointer(config.get("path", "checkpoints.db"))
    if ctype == "pg":
        dsn = config.get("dsn")
        if not isinstance(dsn, str):
            raise ConfigError("checkpoint type 'pg' requires a `dsn`")
        return PGCheckpointer(dsn, config.get("table", "checkpoints"))
    if ctype == "pg_history":
        dsn = config.get("dsn")
        if not isinstance(dsn, str):
            raise ConfigError("checkpoint type 'pg_history' requires a `dsn`")
        return PGHistoryCheckpointer(dsn, config.get("table", "checkpoints"))
    raise ConfigError(
        f"unknown checkpoint type {ctype!r} (file|sqlite|sqlite_history|pg|pg_history)"
    )
