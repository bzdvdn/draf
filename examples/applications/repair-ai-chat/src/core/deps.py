"""Dependency wiring: services, tools and catalog singletons."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from draf.rag.base import VectorStore
from draf.rag.embedder import Embedder
from draf.rag.stores import PGVectorStore, SQLiteVectorStore
from src.domain.services.budget import BudgetService
from src.domain.services.material import MaterialService
from src.domain.services.room import RoomService
from src.rag.catalog import MaterialCatalog

#: Relative to the example root; points at ``data/documents/materials.csv``.
DEFAULT_CATALOG = (
    Path(__file__).resolve().parents[2] / "data" / "documents" / "materials.csv"
)

#: Durable SQLite vector-store path shared by every process (API + workers).
DEFAULT_CATALOG_DB = Path(__file__).resolve().parents[2] / "data" / "catalog.db"

#: The real price list (``Наименование/Цена/Ед/...``), loaded into the RAG
#: alongside the descriptive materials CSV so agents quote real products.
DEFAULT_PRICE_LIST = (
    Path(__file__).resolve().parents[2] / "data" / "documents" / "price.csv"
)

#: ``canonic key -> price.csv column`` for the materialized product rows.
PRODUCT_FIELDMAP = {
    "name": "Наименование",
    "price": "Цена",
    "unit": "Ед",
    "variant": "Вариант",
    "article": "Артикул",
    "brand": "Бренд",
}


def _is_dsn(value: str | Path | None) -> bool:
    """True when *value* is a Postgres connection string (vs a file path)."""
    return str(value or "").startswith(("postgres://", "postgresql://"))


def build_deps(
    catalog_csv: str | Path | None = None,
    *,
    provider: str = "ollama",
    catalog_db: str | Path | None = None,
):
    """Build the plain-Python dependency container.

    *services* is a namespace object: the tools access them as attributes
    (``services.room``, ``services.material``, ...).  Swapping these for
    real databases / vector stores later only touches this module.

    The materials RAG defaults to a durable SQLite vector store at
    ``data/catalog.db`` so every process (API + worker) shares the same
    indexed rows and they survive restarts.  Pass a Postgres DSN as
    *catalog_db* to store the vectors in pgvector instead.
    """
    services = SimpleNamespace(
        room=RoomService(),
        material=MaterialService(),
        budget=BudgetService(),
    )
    store: VectorStore
    if _is_dsn(catalog_db):
        store = PGVectorStore(dsn=str(catalog_db), dim=768)
    else:
        store = SQLiteVectorStore(
            path=str(catalog_db or DEFAULT_CATALOG_DB),
            dim=768,
        )
    catalog = MaterialCatalog(
        embedder=Embedder(provider=provider),
        store=store,
    )
    path = catalog_csv or DEFAULT_CATALOG
    if Path(path).exists():
        catalog.add_csv(str(path))
    if Path(DEFAULT_PRICE_LIST).exists():
        catalog.add_csv(str(DEFAULT_PRICE_LIST), fieldmap=PRODUCT_FIELDMAP)
    # A durable store may already be populated (e.g. by a worker): adopt it.
    catalog.resume()
    return services, catalog
