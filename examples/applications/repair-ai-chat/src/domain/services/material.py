"""Material quantity calculators (tiles, paint, laminate, plaster, putty)."""

import math


class MaterialService:
    @staticmethod
    def calculate_tiles(
        area_m2: float,
        tile_width_cm: float,
        tile_height_cm: float,
        reserve_percent: float = 10.0,
    ) -> dict:
        """Number of tiles plus a cutting/waste reserve."""
        tile_area = (tile_width_cm / 100) * (tile_height_cm / 100)
        pieces = math.ceil(area_m2 / tile_area) if tile_area else 0
        reserve = math.ceil(pieces * reserve_percent / 100)
        return {
            "pieces": pieces,
            "reserve_pieces": reserve,
            "total_pieces": pieces + reserve,
        }

    @staticmethod
    def calculate_paint(
        area_m2: float,
        coats: int = 2,
        consumption_ml_per_m2: float = 100,
    ) -> dict:
        """Paint volume and buckets of 10/5/1 litres."""
        total_liters = math.ceil(area_m2 * coats * consumption_ml_per_m2 / 1000)
        buckets_10 = total_liters // 10
        rest = total_liters % 10
        buckets_5 = rest // 5
        buckets_1 = rest % 5
        return {
            "total_liters": total_liters,
            "buckets_10l": buckets_10,
            "buckets_5l": buckets_5,
            "buckets_1l": buckets_1,
        }

    @staticmethod
    def calculate_laminate(
        area_m2: float,
        pack_coverage_m2: float = 1.98,
        reserve_percent: float = 5,
    ) -> dict:
        """Laminate packs needed for an area with a reserve."""
        area_with_reserve = area_m2 * (1 + reserve_percent / 100)
        packs = math.ceil(area_with_reserve / pack_coverage_m2)
        return {
            "packs": packs,
            "area_with_reserve": round(area_with_reserve, 2),
        }

    @staticmethod
    def calculate_plaster(
        area_m2: float,
        layer_mm: float = 10,
        consumption_kg_per_m2_per_mm: float = 1.5,
    ) -> dict:
        """Plaster weight in kg and 30 kg bags."""
        total_kg = math.ceil(area_m2 * layer_mm * consumption_kg_per_m2_per_mm)
        return {
            "total_kg": total_kg,
            "bags_30kg": math.ceil(total_kg / 30),
        }

    @staticmethod
    def calculate_putty(
        area_m2: float,
        layers: int = 1,
        consumption_kg_per_m2: float = 0.8,
    ) -> dict:
        """Putty weight in kg and 20 kg buckets."""
        total_kg = math.ceil(area_m2 * layers * consumption_kg_per_m2)
        return {
            "total_kg": total_kg,
            "buckets_20kg": math.ceil(total_kg / 20),
        }
