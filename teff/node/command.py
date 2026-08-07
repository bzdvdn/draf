"""Command — a node return value that combines state updates with control flow.

Normally a node returns a plain dict of state updates and the graph routes
along its outgoing edges (``branch`` / ``loop`` / string conditions).
Returning a :class:`Command` lets the node **also** pick the next node —
LangGraph-style dynamic routing::

    class AdminGate(Node):
        async def execute(self, ctx, state):
            if state.get("role") == "admin":
                return Command(update={"allowed": True}, goto="admin_tools")
            return Command(update={"allowed": False}, goto="denied")

* ``update`` — state keys merged after the node (same as returning a plain
  dict, reducers apply).
* ``goto`` — the node id to route to (a dynamic edge, LangGraph-style):
  any node in the graph.  :data:`Command.STOP` terminates the run.

``Command(update=...)`` alone keeps normal edge routing; ``Command()`` is a
no-op result that still lets a node finish without updating state.
"""

from __future__ import annotations

import typing


class Command:
    """Combine a state update with an explicit next-node route.

    Attributes:
        update: State keys merged after the node (same as returning a
            plain dict; per-key reducers apply).
        goto: Node id to route to next — any node in the graph (a dynamic
            edge), or :data:`STOP` to end the run.  ``None`` keeps normal
            edge (condition) routing.
    """

    #: Sentinel for :attr:`goto`: terminate the run from a node.
    STOP: typing.Any = object()

    def __init__(
        self,
        update: dict | None = None,
        goto: str | object | None = None,
    ):
        self.update = dict(update or {})
        self.goto = goto

    def __repr__(self) -> str:
        return f"Command(update={self.update!r}, goto={self.goto!r})"


def as_updates(result: dict | Command) -> dict:
    """Normalise a node result to a plain dict of state updates.

    A plain dict is returned unchanged; a :class:`Command` contributes its
    ``update`` part only (its ``goto`` is a top-level-routing concern and is
    ignored in sub-workflows like ``Parallel``/``Map`` branches).
    """
    return result.update if isinstance(result, Command) else result


__all__ = ["Command", "as_updates"]
