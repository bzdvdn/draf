"""Runtime logging for draf.

Draf ships an operational log stream over the standard :mod:`logging`
module. It answers "what is my workflow doing right now" — the running
chain of nodes, edges and tool calls — without forcing the LLM
prompt/response content into your console.

Levels
------
``INFO``    The full run skeleton: ``run_start`` / ``run_end``, each
            ``node_start`` / ``node_end``, ``edge`` routing, ``llm_call``
            facts (model + token counts, no text) and ``tool_call``.
``DEBUG``   Everything in INFO plus the LLM ``llm_request`` /
            ``llm_response`` **content** (redacted and truncated) and
            ``checkpoint`` activity.
``WARNING`` ``retry``, ``structured``-output validation errors and
            ``interrupt`` pauses.
``ERROR``   ``node_error``.

The distinction keeps INFO a readable "skeleton", while the prompt /
answer visibility is an opt-in debug detail.

Correlation::
=============

Every record is tagged with the enclosing ``run_id`` / ``session_id``
(and, while a node executes, ``node_id`` / ``node_type``) through
:mod:`contextvars`, injected by a :class:`DrafFilter`. Because the ids
live in values rather than the message, that is also sufficient for
structured (JSON) output — the records carry the ids for search and
.post-filtering.

Public API::

    from draf import get_logger, configure_logging

    configure_logging()                         # INFO, text, -> stderr
    configure_logging("debug", format="json")   # content, JSON lines

    log = get_logger("my_app")
    log.info("node_start")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

# Per-call correlation carried through contextvars.  Empty string means
# "unset" (the token is a UUID / short hex for a ``graph.run()``).

_run_id: ContextVar[str] = ContextVar("draf_run_id", default="")
_session_id: ContextVar[str] = ContextVar("draf_session_id", default="")
_node_id: ContextVar[str] = ContextVar("draf_node_id", default="")
_node_type: ContextVar[str] = ContextVar("draf_node_type", default="")

#: Short env var that selectss the default level for
#: :func:`configure_logging` when it is not passed explicitly.
LOG_LEVEL_ENV = "DRAF_LOG_LEVEL"

#: Cap for LLM request/response payloads logged at DEBUG (characters).
LLM_LOG_CHARS_ENV = "DRAF_LOG_LLM_CHARS"

# Standard logging attribute names plus the draf contextvars.  These are
# stripped before carrying the ``extra`` payload into a JSON record.
_RESERVED_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "run_id",
        "session_id",
        "node_id",
        "node_type",
    }
)


def run_id() -> str:
    """Return the current ``run_id`` (empty string outside a run)."""
    return _run_id.get()


def new_run_id() -> str:
    """Return a fresh short run id."""
    return uuid.uuid4().hex[:12]


def get_logger(name: str = "draf") -> logging.Logger:
    """Return a logger for *name*.

    This never configures handlers — configuration is the single concern
    of :func:`configure_logging`.  Use directly in any module::

        log = get_logger(__name__)
        log.info("node_start")
    """
    if not name:
        return logging.getLogger("draf")
    if name == "draf" or name.startswith("draf."):
        return logging.getLogger(name)
    return logging.getLogger(f"draf.{name}")


class _DrafOnlyFilter(logging.Filter):
    """Let only ``draf``-namespaced records reach the handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "draf" or record.name.startswith("draf.")


class ContextFilter(logging.Filter):
    """Attach the current run/session/node ids to every log record.

    The ids come from the :mod:`contextvars` above, so nested or
    concurrent runs do not bleed into one another's records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        record.session_id = _session_id.get()
        record.node_id = _node_id.get()
        record.node_type = _node_type.get()
        return True


class TextFormatter(logging.Formatter):
    """Human-readable ``[run=.. session=.. node=.. type=..]`` header."""

    _PREFIX = "%(asctime)s %(levelname)-8s %(name)-20s "

    def format(self, record: logging.LogRecord) -> str:
        header = [
            ("run", getattr(record, "run_id", "")),
            ("session", getattr(record, "session_id", "")),
            ("node", getattr(record, "node_id", "")),
            ("type", getattr(record, "node_type", "")),
        ]
        ctx = " ".join(f"{k}={v}" for k, v in header if v)
        filled = dict(record.__dict__)
        filled["asctime"] = self.formatTime(record, self.datefmt)
        prefix = self._PREFIX % filled
        middle = f"[{ctx}] " if ctx else ""
        return f"{prefix}{middle}{record.getMessage()}".rstrip()


class JsonFormatter(logging.Formatter):
    """Single-line JSON log records (secrets redacted where possible)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _fmt(record.created),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "run_id": getattr(record, "run_id", "") or None,
            "session_id": getattr(record, "session_id", "") or None,
            "node_id": getattr(record, "node_id", "") or None,
            "node_type": getattr(record, "node_type", "") or None,
        }
        for key in record.__dict__:
            if key in _RESERVED_ATTRS:
                continue
            payload[key] = getattr(record, key)
        return json.dumps(payload, default=str, ensure_ascii=False)


def _fmt(ts: float) -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def configure_logging(level: int | str | None = None, format: str = "text") -> None:
    """Configure the root logger for draf.

    Args:
        level: One of the ``logging`` thresholds (``INFO`` …) or a
            string name.  When ``None``, read ``DRAF_LOG_LEVEL`` env var,
            defaulting to ``INFO``.
        format: ``"text"`` (default, human-readable, stderr) or
            ``"json"`` (single-line JSON per record, stdout).

    Output file:
        ``text`` goes to ``stderr``; ``json`` goes to ``stdout``.  This
        keeps human diagnostics off the pipe where a caller streams JSON
        results.
    """
    if level is None:
        level = os.environ.get(LOG_LEVEL_ENV, "INFO")
    root = logging.getLogger()
    root.setLevel(level)

    # Idempotent: reuse an existing draf handler rather than stacking a
    # new one on every call (e.g. when an app calls configure twice).
    # Switching formats swaps the whole handler so the stream (stderr for
    # text, stdout for JSON) matches the requested format.
    handler = next(
        (h for h in root.handlers if getattr(h, "_draf_handler", False)),
        None,
    )
    if handler is not None and getattr(handler, "_draf_format", None) != format:
        root.removeHandler(handler)
        handler = None
    if handler is None:
        handler = _build_handler(format)
        handler._draf_handler = True  # type: ignore[attr-defined]
        handler._draf_format = format  # type: ignore[attr-defined]
        handler.addFilter(ContextFilter())
        handler.addFilter(_DrafOnlyFilter())
        root.addHandler(handler)
    handler.setLevel(level)
    handler.setFormatter(_build_formatter(format))
    root.propagate = False


def _build_handler(format: str) -> logging.Handler:
    if format == "json":
        return logging.StreamHandler(sys.stdout)
    return logging.StreamHandler()


def _build_formatter(format: str) -> logging.Formatter:
    if format == "json":
        return JsonFormatter()
    return TextFormatter()


@contextmanager
def run_id_ctx(*, run_id: str = "", session_id: str = "") -> Any:
    """Set ``run_id`` / ``session_id`` for the entered block.

    Restores the previous values on exit; designed for ``graph.run()``
    and ``graph.stream()``.
    """
    prev_run = _run_id.set(run_id)
    prev_session = _session_id.set(session_id)
    try:
        yield
    finally:
        _run_id.reset(prev_run)
        _session_id.reset(prev_session)


@contextmanager
def node_id_ctx(*, node_id: str = "", node_type: str = "") -> Iterator:
    """Set ``node_id`` / ``node_type`` for the entered block."""
    prev_node = _node_id.set(node_id)
    prev_type = _node_type.set(node_type)
    try:
        yield
    finally:
        _node_id.reset(prev_node)
        _node_type.reset(prev_type)
