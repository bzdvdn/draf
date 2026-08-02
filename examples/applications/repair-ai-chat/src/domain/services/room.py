"""Room geometry calculations."""


class RoomService:
    @staticmethod
    def wall_area(length: float, width: float, height: float) -> float:
        """Wall area from a room's length, width, and ceiling height."""
        return round(2 * (length + width) * height, 2)

    @staticmethod
    def floor_area(length: float, width: float) -> float:
        """Floor area of a rectangular room."""
        return round(length * width, 2)

    @staticmethod
    def ceiling_area(length: float, width: float) -> float:
        """Ceiling area of a rectangular room."""
        return round(length * width, 2)

    @staticmethod
    def perimeter(length: float, width: float) -> float:
        """Perimeter of a rectangular room."""
        return round(2 * (length + width), 2)
