"""Budget/estimate tools."""

from draf.tool.tool import Tool

from src.domain.services.budget import BudgetService


class EstimateMaterialCost(Tool):
    name = "estimate_material_cost"
    description = "Calculate the cost of a single material line item."

    def __init__(self, service: BudgetService):
        super().__init__()
        self.service = service

    def run(self, name: str, quantity: float, unit_price: float) -> dict:  # type: ignore[override]
        total = self.service.calculate_item(quantity, unit_price)
        return {
            "name": name,
            "quantity": quantity,
            "unit_price": unit_price,
            "total": total,
        }


class EstimateTotal(Tool):
    name = "estimate_total"
    description = "Calculate total estimate including labor and overhead."

    def __init__(self, service: BudgetService):
        super().__init__()
        self.service = service

    def run(  # type: ignore[override]
        self,
        items_total: float,
        labor_cost: float = 0,
        overhead_percent: float = 10,
    ) -> dict:
        return self.service.calculate_total(items_total, labor_cost, overhead_percent)
