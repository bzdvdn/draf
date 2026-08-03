"""Room geometry tools."""

from draf.tool.tool import Tool
from src.domain.services.room import RoomService


class CalculateWallArea(Tool):
    name = "calculate_wall_area"
    description = (
        "Calculate the total wall area of a room. "
        "Useful for estimating paint, wallpaper, or plaster."
    )

    def __init__(self, service: RoomService):
        super().__init__()
        self.service = service

    def run(self, length: float, width: float, height: float) -> float:  # type: ignore[override]
        return self.service.wall_area(length, width, height)


class CalculateFloorArea(Tool):
    name = "calculate_floor_area"
    description = (
        "Calculate the floor area of a room. Useful for tiles, laminate, or flooring."
    )

    def __init__(self, service: RoomService):
        super().__init__()
        self.service = service

    def run(self, length: float, width: float) -> float:  # type: ignore[override]
        return self.service.floor_area(length, width)


class CalculateCeilingArea(Tool):
    name = "calculate_ceiling_area"
    description = "Calculate the ceiling area of a room."

    def __init__(self, service: RoomService):
        super().__init__()
        self.service = service

    def run(self, length: float, width: float) -> float:  # type: ignore[override]
        return self.service.ceiling_area(length, width)


class CalculatePerimeter(Tool):
    name = "calculate_perimeter"
    description = (
        "Calculate the perimeter of a room. Useful for baseboards or crown molding."
    )

    def __init__(self, service: RoomService):
        super().__init__()
        self.service = service

    def run(self, length: float, width: float) -> float:  # type: ignore[override]
        return self.service.perimeter(length, width)
