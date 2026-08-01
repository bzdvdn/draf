"""Dependency wiring: services, tools and catalog singletons."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from draf.rag.embedder import Embedder

from src.domain.services.budget import BudgetService
from src.domain.services.material import MaterialService
from src.domain.services.room import RoomService
from src.rag.catalog import MaterialCatalog

#: Relative to the example root; points at ``data/documents/materials.csv``.
DEFAULT_CATALOG = (
    Path(__file__).resolve().parents[2] / "data" / "documents" / "materials.csv"
)


def build_deps(catalog_csv: str | Path | None = None, *, provider: str = "ollama"):
    """Build the plain-Python dependency container.

    *services* is a namespace object: the tools access them as attributes
    (``services.room``, ``services.material``, ...).  Swapping these for
    real databases / vector stores later only touches this module.
    """
    services = SimpleNamespace(
        room=RoomService(),
        material=MaterialService(),
        budget=BudgetService(),
    )
    catalog = MaterialCatalog(embedder=Embedder(provider=provider))
    path = catalog_csv or DEFAULT_CATALOG
    if Path(path).exists():
        catalog.add_csv(str(path))
    return services, catalog
