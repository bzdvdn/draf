"""Public exception hierarchy for draf.

Everything the framework raises derives from :class:`DrafError`, so a
single ``except draf.DrafError`` is enough to catch any library error,
while specific subclasses let callers branch on the failure mode.

The hierarchy deliberately multiple-inherits from builtin exceptions to
stay backwards-compatible: code that already does ``except KeyError``,
``except ValueError`` or ``except RuntimeError`` keeps working.

    DrafError
    ├── ConfigError        (also KeyError) — invalid config / unknown types
    ├── WorkflowError      (also RuntimeError) — workflow-level failures
    │   ├── NodeError      — a node raised (carries ``node_id``/``node_type``)
    │   └── LLMError       — a model call failed after retries/fallbacks
    ├── InterruptError     — HITL resume misuse
    ├── GraphInterrupt     — workflow paused for human input
    └── StructuredOutputError (also ValueError) — schema validation failed

Note: transport-level failures (timeouts, HTTP status errors, connection
errors) propagate as the underlying ``httpx`` exceptions so that existing
``except httpx.XError`` handlers keep working.
"""

from __future__ import annotations

import re
from typing import Any

_SECRET_KEYS = (
    "api[_-]?key",
    "apikey",
    "authorization",
    "auth",
    "token",
    "secret",
    "password",
    "access[_-]?key",
    "x-api-key",
)

_SECRET_KEY_RE = re.compile(
    r"("
    r'"?(' + "|".join(_SECRET_KEYS) + r')"?\s*[:=]\s*'
    r')("?)(?:[^\s,;&"]+\s+)*[^\s,;&"]{6,}\3',
    re.IGNORECASE,
)


class DrafError(Exception):
    """Base class for every exception raised by the draf framework."""


class ConfigError(DrafError, KeyError):
    """Invalid configuration: bad workflow YAML, unknown node/tool type.

    Subclasses :class:`KeyError` so legacy ``except KeyError`` blocks
    continue to catch unknown-type lookups.
    """


class WorkflowError(DrafError, RuntimeError):
    """A workflow failed at runtime (loop guards, execution invariants).

    Subclasses :class:`RuntimeError` so legacy ``except RuntimeError``
    blocks (e.g. ``max_iterations``) keep working.
    """


class NodeError(WorkflowError):
    """A node failed; carries the failing ``node_id`` and ``node_type``."""

    def __init__(
        self,
        message: str,
        *,
        node_id: str | None = None,
        node_type: str | None = None,
    ):
        super().__init__(message)
        self.node_id = node_id
        self.node_type = node_type


class LLMError(WorkflowError):
    """A model call failed after exhausting retries and fallbacks.

    The original transport exception is available as ``__cause__``.
    """


class InterruptError(DrafError):
    """Raised when an interrupt/resume contract is violated.

    For example: resuming a run that has no pending interrupt, or
    resuming without the checkpoint that holds the pause.
    """


def as_node_error(
    exc: Exception, *, node_id: str | None, node_type: str | None
) -> NodeError:
    """Wrap *exc* into a :class:`NodeError` carrying node context."""
    error = NodeError(
        f"{node_type or 'node'} '{node_id}' failed: {exc}",
        node_id=node_id,
        node_type=node_type,
    )
    error.__cause__ = exc
    return error


def redact(value: Any, keys: tuple[str, ...] = ()) -> Any:
    """Mask secret-looking values in *value* (recursively).

    - dict values whose (lowercased) key names a secret are replaced with
      ``"***"`` — this catches ``{"Authorization": "Bearer sk-..."}``.
    - string values are scanned for ``key=...`` / ``key: ...`` pairs that
      name a secret (e.g. ``?api_key=sk-abc`` inside a URL) and the value
      is masked in place.

    *keys* is the word-list used for the string regex; when empty a
    sensible default list is used.
    """
    if isinstance(value, dict):
        keyset = {k.lower() for k in keys} if keys else _default_keys
        return {
            k: redact(v, keys) if k.lower() not in keyset else "***"
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v, keys) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v, keys) for v in value)
    if isinstance(value, str):
        if not keys:
            return _SECRET_KEY_RE.sub(r"\1***", value)
        pattern = re.compile(
            r'("?'
            + "|".join(re.escape(k) for k in keys)
            + r'"?\s*[:=]\s*)("?)(?:[^\s,;&"]+\s+)*[^\s,;&"]{6,}\2',
            re.IGNORECASE,
        )
        return pattern.sub(r"\1***", value)
    return value


_default_keys = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "auth",
        "token",
        "access_token",
        "secret",
        "password",
        "x-api-key",
    }
)
