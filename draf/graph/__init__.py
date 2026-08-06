"""Graph data structure for representing agent workflows.

The package is split into focused modules:

- :mod:`draf.graph.edge` — the :class:`~draf.graph.Edge` model and hooks.
- :mod:`draf.graph.conditions` — edge condition evaluation.
- :mod:`draf.graph.execution` — the execution engine behind ``run()``.
- :mod:`draf.graph.render` — Mermaid / YAML serialization.
- :mod:`draf.graph.graph` — the :class:`~draf.graph.Graph` facade.

``Edge``, ``Graph``, and ``Hook`` are re-exported here for convenience.
"""

from draf.graph.edge import _ERROR_CONDITION, _INTERRUPT_KEY, Edge, Hook
from draf.graph.graph import Graph, TurnResult

__all__ = [
    "Edge",
    "Graph",
    "Hook",
    "TurnResult",
    "_ERROR_CONDITION",
    "_INTERRUPT_KEY",
]
