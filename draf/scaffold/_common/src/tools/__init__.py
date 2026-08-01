"""Tool registry — build the tool set handed to the graph.

Tools are plain ``draf.Tool`` subclasses.  Instantiate them here (once) so
``graphs/build.py`` stays free of wiring details.  When a tool needs a
service or a database, construct it here and pass it in — everything that
touches the outside world stays in this module.
"""

from __future__ import annotations

from src.tools.example import CurrentDate


def build_tools() -> list:
    """Build the full tool set for :func:`src.graphs.build.build_flow`."""
    return [
        CurrentDate(),
    ]
