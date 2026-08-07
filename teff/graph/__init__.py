"""Graph data structure for representing agent workflows.

The package is split into focused modules:

- :mod:`teff.graph.edge` — the :class:`~teff.graph.Edge` model and hooks.
- :mod:`teff.graph.conditions` — edge condition evaluation.
- :mod:`teff.graph.execution` — the execution engine behind ``run()``.
- :mod:`teff.graph.render` — Mermaid / YAML serialization.
- :mod:`teff.graph.graph` — the :class:`~teff.graph.Graph` facade.

``Edge``, ``Graph``, and ``Hook`` are re-exported here for convenience.
"""

from teff.graph.edge import _ERROR_CONDITION, _INTERRUPT_KEY, Edge, Hook
from teff.graph.graph import Graph, TurnResult

__all__ = [
    "Edge",
    "Graph",
    "Hook",
    "TurnResult",
    "_ERROR_CONDITION",
    "_INTERRUPT_KEY",
]
