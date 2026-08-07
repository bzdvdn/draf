"""Declarative ``command`` node — route the graph from YAML state.

The :class:`CommandNode` is the YAML surface for :class:`Command` routing
(``goto`` / ``STOP``).  It lives in its own module so that
:mod:`teff.node.command` (imported by the base :class:`~teff.node.node.Node`)
never needs to import the node base class itself — avoiding a circular
import.
"""

from __future__ import annotations

from teff.node.command import Command
from teff.node.node import Node


class CommandNode(Node):
    """Declarative ``command`` node: route the graph from YAML state.

    Returns a :class:`~teff.node.command.Command` whose ``goto`` is chosen
    from ``routes`` (the first route whose ``when`` condition matches
    *state*, using the same expressions as ``edges:`` conditions) and falls
    back to ``goto``.  ``update`` merges state keys after routing (reducers
    apply).

    Use ``STOP`` as a target to terminate the run::

        - id: route
          type: command
          config:
            routes:
              - when: score >= 0.8
                goto: approve
              - when: score < 0.3
                goto: reject
            goto: review
            update: {routed: true}
    """

    type = "command"

    async def execute(self, ctx, state: dict) -> Command:
        from teff.graph.conditions import evaluate

        goto: str | object | None = None
        for route in self.config.get("routes", []) or []:
            when = route.get("when")
            if when and evaluate(when, state):
                goto = _resolve_target(route.get("goto"))
                break
        if goto is None:
            goto = _resolve_target(self.config.get("goto"))
        return Command(update=dict(self.config.get("update") or {}), goto=goto)


def _resolve_target(target: str | None) -> str | object | None:
    """Map the ``STOP`` string in YAML to the :attr:`Command.STOP` sentinel."""
    if target is None:
        return None
    if target == "STOP":
        return Command.STOP
    return target


__all__ = ["CommandNode"]
