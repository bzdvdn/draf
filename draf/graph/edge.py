"""Graph edge model and hook protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from draf.node.node import Node

#: Condition that matches when a node execution raises an exception.
_ERROR_CONDITION = "__error__"
#: Internal state key that carries a paused interrupt payload.
_INTERRUPT_KEY = "__interrupt__"


@dataclass
class Edge:
    """A directed edge between two nodes with an optional condition.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        condition: Expression ``key=value``, ``key!=value``,
            comma-separated disjunction ``key=a,b``, or numeric comparison
            ``key>=N`` / ``key<=N`` / ``key>N`` / ``key<N``.
            ``None`` means unconditional.
            ``"__error__"`` matches when the source node raises an exception.
    """

    source_id: str
    target_id: str
    condition: str | None = None


Hook = Callable[[str, Node, dict], Any]
"""Signature for observability hooks: ``(node_id, node, state)``."""

__all__ = ["Edge", "Hook", "_ERROR_CONDITION", "_INTERRUPT_KEY"]
