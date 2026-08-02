"""Tool registry — build the tool set handed to the graph.

Tools are plain ``draf.Tool`` subclasses.  Instantiate them here (once) so
``graphs/build.py`` stays free of wiring details.  When a tool needs a
service or a database, construct it here and pass it in — everything that
touches the outside world stays in this module.

The ``rag`` variant adds a catalog-backed search tool: pass the catalog
built by ``src.core.container`` into :func:`build_tools` and the RAG tools
are appended to the pool.
"""

from __future__ import annotations

from src.tools.example import CurrentDate


def _rag_tools(catalog) -> list:
    """RAG search tools for the ``rag`` variant (no-op without a catalog)."""
    if catalog is None:
        return []
    from src.tools.rag import FindSimilar, SearchCatalog

    return [SearchCatalog(catalog), FindSimilar(catalog)]


def build_tools(catalog=None) -> list:
    """Build the full tool set for :func:`src.graphs.build.build_flow`.

    Args:
        catalog: The ``src.rag`` catalog (``rag`` variant); when given, the
            RAG search tools are appended to the pool.
    """
    return [CurrentDate(), *_rag_tools(catalog)]
