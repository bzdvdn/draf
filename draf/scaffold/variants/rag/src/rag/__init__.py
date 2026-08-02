"""RAG variant — document catalog + wiring.

This module only exists in projects scaffolded with ``--with rag``.  The
base composition root (:func:`src.core.build_container`) imports it
conditionally, so plain projects carry no RAG code.  Building a catalog
never touches the network: documents are embedded lazily on the first
search.
"""

from src.rag.catalog import DocumentCatalog, IngestReport
from src.rag.wiring import DEFAULT_DOCUMENTS, build_catalog

__all__ = [
    "DocumentCatalog",
    "IngestReport",
    "DEFAULT_DOCUMENTS",
    "build_catalog",
]
