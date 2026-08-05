"""Build a :class:`GraphObserver` from a workflow's ``observability:`` block.

The block is declarative data inside a ``workflow.yaml``::

    observability:
      db: ./data/traces.db          # our SQLite dashboard (relative to the workflow)
      export:                       # optional fan-out to remote sinks
        - type: langfuse
          public_key_env: LANGFUSE_PUBLIC_KEY
          secret_key_env: LANGFUSE_SECRET_KEY
          host: https://cloud.langfuse.com
        - type: langsmith
          api_key_env: LANGCHAIN_API_KEY
          project: my-project
        - type: webhook
          url: https://hooks.example.com/traces

Relative paths resolve against the workflow file's directory and secrets
come from environment variables, never from the file itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from draf.errors import ConfigError
from draf.observability.collector import GraphObserver
from draf.observability.exporter import CompositeExporter, SQLiteExporter, TraceExporter
from draf.observability.push import HttpExporter, LangfuseExporter, LangsmithExporter
from draf.observability.topology import topology_from_graph


def _resolve_path(base_dir: str, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path)


def _require_env(spec: dict[str, Any], key: str, default_env: str, kind: str) -> str:
    name = spec.get(key) or default_env
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"observability.export.{kind}: env var {name!r} is required")
    return value


def build_remote_exporter(
    spec: dict[str, Any], *, base_dir: str = "."
) -> TraceExporter:
    """Construct a remote exporter from one ``observability.export`` entry."""
    kind = spec.get("type")
    timeout = float(spec.get("timeout", 10.0))
    retries = int(spec.get("retries", 3))
    backoff = float(spec.get("backoff", 1.0))
    if kind == "webhook":
        url = spec.get("url")
        if not url and spec.get("url_env"):
            url = os.environ.get(spec["url_env"])
        if not url:
            raise ConfigError(
                "observability.export.webhook: 'url' or 'url_env' is required"
            )
        return HttpExporter(
            url,
            headers=dict(spec.get("headers") or {}),
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
    if kind == "langfuse":
        host = spec.get("host")
        if not host:
            raise ConfigError("observability.export.langfuse: 'host' is required")
        public_key = _require_env(
            spec, "public_key_env", "LANGFUSE_PUBLIC_KEY", "langfuse"
        )
        secret_key = _require_env(
            spec, "secret_key_env", "LANGFUSE_SECRET_KEY", "langfuse"
        )
        return LangfuseExporter(
            host,
            public_key,
            secret_key,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
    if kind == "langsmith":
        api_url = (
            spec.get("api_url")
            or os.environ.get("LANGCHAIN_ENDPOINT")
            or "https://api.smith.langchain.com"
        )
        api_key = _require_env(spec, "api_key_env", "LANGCHAIN_API_KEY", "langsmith")
        project = spec.get("project") or os.environ.get("LANGCHAIN_PROJECT")
        return LangsmithExporter(
            api_url,
            api_key,
            project=project,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
    raise ConfigError(f"observability.export: unknown exporter type {kind!r}")


def _build_exporters(config: dict[str, Any], *, base_dir: str) -> list[TraceExporter]:
    """Construct all sinks declared in an ``observability:`` block."""
    exporters: list[TraceExporter] = []
    db = config.get("db")
    if db:
        exporters.append(SQLiteExporter(_resolve_path(base_dir, str(db))))
    for spec in config.get("export") or []:
        exporters.append(build_remote_exporter(spec, base_dir=base_dir))
    return exporters


def build_observability(
    config: dict[str, Any] | None,
    *,
    base_dir: str = ".",
    graph=None,
    name: str = "workflow",
) -> GraphObserver | None:
    """Assemble a :class:`GraphObserver` from an ``observability:`` block.

    Returns ``None`` when the block is missing or declares no sinks.  The
    observer is wired with the graph topology so remote exporters and the
    dashboard can render the flow.
    """
    if not config:
        return None
    exporters = _build_exporters(config, base_dir=base_dir)
    if not exporters:
        return None
    return GraphObserver(
        name,
        exporter=CompositeExporter(exporters),
        topology=topology_from_graph(graph) if graph is not None else None,
    )


def build_observer_factory(
    config: dict[str, Any] | None,
    *,
    base_dir: str = ".",
    graph=None,
    name: str = "workflow",
) -> "Callable[[], GraphObserver] | None":
    """Like :func:`build_observability` but for repeated runs.

    Returns a zero-arg callable that yields a *fresh* observer per run while
    sharing one set of exporters — the right shape for a daemon that traces
    every tick.  ``None`` when observability is not configured.
    """
    if not config:
        return None
    exporters = _build_exporters(config, base_dir=base_dir)
    if not exporters:
        return None
    composite = CompositeExporter(exporters)
    topology = topology_from_graph(graph) if graph is not None else None

    def factory() -> GraphObserver:
        return GraphObserver(name, exporter=composite, topology=topology)

    return factory
