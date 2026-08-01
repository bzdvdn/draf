"""Estimate/budget calculations."""


class BudgetService:
    @staticmethod
    def calculate_item(quantity: float, unit_price: float) -> float:
        """Cost of a single line item."""
        return round(quantity * unit_price, 2)

    @staticmethod
    def calculate_total(
        items_total: float,
        labor_cost: float = 0,
        overhead_percent: float = 10,
    ) -> dict:
        """Estimate total including labor and overhead."""
        overhead = round((items_total + labor_cost) * overhead_percent / 100, 2)
        total = round(items_total + labor_cost + overhead, 2)
        return {
            "items_total": items_total,
            "labor_cost": labor_cost,
            "overhead_percent": overhead_percent,
            "overhead_amount": overhead,
            "total": total,
        }
