"""Material quantity tools."""

from src.domain.services.material import MaterialService
from teff.tool.tool import Tool


class CalculateTiles(Tool):
    name = "calculate_tiles"
    description = "Calculate the number of tiles needed for a given area."

    def __init__(self, service: MaterialService):
        super().__init__()
        self.service = service

    def run(  # type: ignore[override]
        self,
        area_m2: float,
        tile_width_cm: float,
        tile_height_cm: float,
        reserve_percent: float = 10,
    ) -> dict:
        return self.service.calculate_tiles(
            area_m2, tile_width_cm, tile_height_cm, reserve_percent
        )


class CalculatePaint(Tool):
    name = "calculate_paint"
    description = "Calculate the amount of paint needed."

    def __init__(self, service: MaterialService):
        super().__init__()
        self.service = service

    def run(  # type: ignore[override]
        self,
        area_m2: float,
        coats: int = 2,
        consumption_ml_per_m2: float = 100,
    ) -> dict:
        return self.service.calculate_paint(area_m2, coats, consumption_ml_per_m2)


class CalculateLaminate(Tool):
    name = "calculate_laminate"
    description = "Calculate the number of laminate packs needed."

    def __init__(self, service: MaterialService):
        super().__init__()
        self.service = service

    def run(  # type: ignore[override]
        self,
        area_m2: float,
        pack_coverage_m2: float = 1.98,
        reserve_percent: float = 5,
    ) -> dict:
        return self.service.calculate_laminate(
            area_m2, pack_coverage_m2, reserve_percent
        )


class CalculatePlaster(Tool):
    name = "calculate_plaster"
    description = "Calculate the amount of plaster needed."

    def __init__(self, service: MaterialService):
        super().__init__()
        self.service = service

    def run(  # type: ignore[override]
        self,
        area_m2: float,
        layer_mm: float = 10,
        consumption_kg_per_m2_per_mm: float = 1.5,
    ) -> dict:
        return self.service.calculate_plaster(
            area_m2, layer_mm, consumption_kg_per_m2_per_mm
        )


class CalculatePutty(Tool):
    name = "calculate_putty"
    description = "Calculate the amount of putty needed."

    def __init__(self, service: MaterialService):
        super().__init__()
        self.service = service

    def run(  # type: ignore[override]
        self,
        area_m2: float,
        layers: int = 1,
        consumption_kg_per_m2: float = 0.8,
    ) -> dict:
        return self.service.calculate_putty(area_m2, layers, consumption_kg_per_m2)
