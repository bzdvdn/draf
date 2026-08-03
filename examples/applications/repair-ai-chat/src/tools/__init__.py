"""Tool registry for the repair agents.

Every tool is a :class:`~draf.tool.Tool` subclass; the set an agent may
call is narrowed per-agent via ``use_tools`` in the graph builder.
"""

from src.tools.budget import EstimateMaterialCost, EstimateTotal
from src.tools.material import (
    CalculateLaminate,
    CalculatePaint,
    CalculatePlaster,
    CalculatePutty,
    CalculateTiles,
)
from src.tools.rag import FindSimilarMaterial, SearchMaterials
from src.tools.room import (
    CalculateCeilingArea,
    CalculateFloorArea,
    CalculatePerimeter,
    CalculateWallArea,
)


def build_tools(services, catalog) -> list:
    """Instantiate the full tool set bound to *services* and *catalog*."""
    return [
        CalculateWallArea(services.room),
        CalculateFloorArea(services.room),
        CalculateCeilingArea(services.room),
        CalculatePerimeter(services.room),
        CalculateTiles(services.material),
        CalculatePaint(services.material),
        CalculateLaminate(services.material),
        CalculatePlaster(services.material),
        CalculatePutty(services.material),
        EstimateMaterialCost(services.budget),
        EstimateTotal(services.budget),
        SearchMaterials(catalog),
        FindSimilarMaterial(catalog),
    ]


__all__ = ["build_tools"]
