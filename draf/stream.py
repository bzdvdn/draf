"""Streaming events for graph execution.

Constitution Principle IX: observability is mandatory.  ``graph.stream()``
emits :class:`StreamEvent` objects as a run progresses, so callers can
render tokens, progress, and routing decisions before the run finishes —
instead of waiting for a fully materialised result from ``graph.run()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    """A single event emitted while a graph streams.

    Attributes:
        type: Event type — ``run_start``, ``node_start``, ``node_end``,
            ``node_error``, ``edge``, ``token``, ``llm``, ``interrupt``,
            ``interrupt_resume``, ``checkpoint``, or ``run_end``.
        node_id: Graph node id the event belongs to, if any.
        node_type: Node type string, if any.
        data: Type-specific payload (token text, error, condition, etc.).
    """

    type: str
    node_id: str | None = None
    node_type: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
