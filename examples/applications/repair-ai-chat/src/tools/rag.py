"""RAG tools backed by the materials catalog."""

from draf.tool.tool import Tool


class SearchMaterials(Tool):
    name = "search_materials"
    description = (
        "Search materials catalog by description or requirements. "
        "Useful when user asks about specific materials, prices, or alternatives."
    )

    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

    async def arun(  # type: ignore[override]
        self,
        query: str,
        category: str | None = None,
        max_price: float | None = None,
    ) -> str:
        return await self.catalog.search(query, category=category, max_price=max_price)


class FindSimilarMaterial(Tool):
    name = "find_similar_material"
    description = (
        "Find materials similar to the given name or description. "
        "Useful when user asks for analogues or alternatives."
    )

    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

    async def arun(self, name: str, top_k: int = 3) -> str:  # type: ignore[override]
        return await self.catalog.find_similar(name, top_k=top_k)
