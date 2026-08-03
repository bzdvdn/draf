"""Folder-based plugin discovery for custom nodes and tools.

A plugin is any Python module that registers node types and/or tools via
the existing ``@node`` / ``@tool`` decorators.  Loading a plugin simply
imports the file — registration happens as a side effect of the import,
so once loaded the new types are visible to ``validate`` / ``load_workflow``
because both read from the shared registries.

Plugins are discovered from the ``plugins`` key of a workflow YAML (paths
to ``.py`` files or directories) and, by default, from a ``plugins/``
folder next to the workflow file::

    plugins:
      - ./custom_nodes.py
      - ./vendor/tools
    steps:
      - id: greet
        type: my_custom_node       # registered by custom_nodes.py
"""

import importlib.util
import os
import re
import sys

from draf.errors import ConfigError

_loaded: set[str] = set()


def _module_name(path: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", path)
    return f"_draf_plugin_{stem}"


def load_plugin_file(path: str) -> None:
    """Import a single plugin ``.py`` file (idempotent per path)."""
    path = os.path.abspath(path)
    if path in _loaded:
        return
    if not os.path.isfile(path):
        raise ConfigError(f"plugin file not found: {path}")
    _loaded.add(path)
    module_name = _module_name(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot load plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"failed to load plugin {path}: {exc}") from exc


def load_plugin_dir(path: str) -> list[str]:
    """Import every ``*.py`` file under *path* (non-recursive).

    Files starting with ``_`` are skipped so ``__init__.py`` and helpers
    are ignored.  Returns the list of imported file paths.
    """
    if not os.path.isdir(path):
        raise ConfigError(f"plugin directory not found: {path}")
    loaded: list[str] = []
    for p in sorted(os.listdir(path)):
        if not p.endswith(".py") or p.startswith("_"):
            continue
        load_plugin_file(os.path.join(path, p))
        loaded.append(os.path.abspath(os.path.join(path, p)))
    return loaded


def load_plugins(
    entries: str | list[str] | None,
    base_dir: str = ".",
    default_folder: str = "plugins",
) -> list[str]:
    """Load plugins from explicit *entries* plus a default folder.

    Each entry is a path — relative to *base_dir*, or absolute — to a
    ``.py`` file or a directory of ``.py`` files.  The *default_folder*
    (default ``"plugins"``) is always loaded when it exists, so a user
    can drop a ``plugins/`` folder next to a workflow with no YAML
    changes.

    Returns the list of loaded file paths.
    """
    if isinstance(entries, str):
        entries = [entries]
    loaded: list[str] = []
    for entry in entries or []:
        p = entry if os.path.isabs(entry) else os.path.join(base_dir, entry)
        if os.path.isdir(p):
            loaded.extend(load_plugin_dir(p))
        else:
            load_plugin_file(p)
            loaded.append(p)
    folder = (
        default_folder
        if os.path.isabs(default_folder)
        else os.path.join(base_dir, default_folder)
    )
    if os.path.isdir(folder):
        loaded.extend(load_plugin_dir(folder))
    return loaded


def load_plugins_from_document(data: dict, base_dir: str) -> list[str]:
    """Load plugins declared in a workflow document.

    Reads ``data["plugins"]`` (files/folders) and ``data["plugins_folder"]``
    (a single folder, defaulting to ``"plugins"``) and loads them relative
    to *base_dir*.
    """
    folder = data.get("plugins_folder", "plugins")
    return load_plugins(data.get("plugins"), base_dir, default_folder=folder)


def reset_plugins() -> None:
    """Clear the import cache (used by tests)."""
    _loaded.clear()
