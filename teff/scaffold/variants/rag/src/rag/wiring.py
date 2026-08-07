"""Wiring for the ``rag`` variant — build the catalog from settings.

Kept separate from ``catalog.py`` so the catalog stays a pure data
structure and all environment/store decisions live here.  The base
composition root imports :func:`build_catalog` (guarded), so this module is
the only rag-specific import a project ever needs.
"""

from __future__ import annotations

from pathlib import Path

from src.config.config import Settings, get_settings
from src.rag.catalog import DocumentCatalog

from teff.rag.embedder import Embedder
from teff.rag.stores import PGVectorStore, SQLiteVectorStore

#: Where the catalog looks for documents (relative to the project root).
DEFAULT_DOCUMENTS = Path(__file__).resolve().parents[2] / "data" / "documents"

#: Durable SQLite vector-store path shared by every process (API + workers).
DEFAULT_VECTORS = Path(__file__).resolve().parents[2] / "data" / "vectors.db"

#: Embedding dimension used by the default stores.
VECTOR_DIM = 768

#: File suffixes the catalog indexes from ``data/documents/``.
_DOCUMENT_SUFFIXES = {".txt", ".md", ".csv"}


def _is_dsn(value: str | None) -> bool:
    """True when *value* is a Postgres connection string (vs a file path)."""
    return str(value or "").startswith(("postgres://", "postgresql://"))


def _build_store(settings: Settings):
    """Pick the vector store: pgvector when a DSN is set, else SQLite."""
    if _is_dsn(settings.database_url):
        return PGVectorStore(dsn=str(settings.database_url), dim=VECTOR_DIM)
    return SQLiteVectorStore(path=str(DEFAULT_VECTORS), dim=VECTOR_DIM)


def build_catalog(settings: Settings | None = None) -> DocumentCatalog:
    """Build the document catalog from *settings* (environment by default).

    Never touches the network: documents are embedded lazily on the first
    search.  A durable store that is already populated (e.g. indexed by a
    Celery worker) is adopted via :meth:`DocumentCatalog.resume`, so every
    process shares the same index.
    """
    settings = settings or get_settings()
    catalog = DocumentCatalog(
        embedder=Embedder(provider=settings.rag_embedder or settings.provider),
        store=_build_store(settings),
        top_k=settings.rag_top_k,
    )
    if DEFAULT_DOCUMENTS.is_dir():
        for path in sorted(DEFAULT_DOCUMENTS.iterdir()):
            if path.is_file() and path.suffix.lower() in _DOCUMENT_SUFFIXES:
                catalog.add_file(str(path))
    catalog.resume()
    return catalog
