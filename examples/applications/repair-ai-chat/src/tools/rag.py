"""RAG tools backed by the materials catalog."""

from draf.tool.tool import Tool


class SearchMaterials(Tool):
    name = "search_materials"
    description = (
        "Search materials catalog by description or requirements. "
        "Useful when user asks about specific materials, prices, or alternatives. "
        "category must be a catalog category, not a room type "
        "(плитка, краска, ламинат, шпаклёвка, штукатурка, грунтовка, гипсокартон, паркет)."
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
        "Useful when user asks for analogues or alternatives. "
        "Pass the material name as 'materials' (a list) or 'name' (a string). "
        "category is optional and must be a catalog category "
        "(плитка, краска, ламинат, шпаклёвка, штукатурка, грунтовка, гипсокартон, паркет)."
    )

    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

    async def arun(  # type: ignore[override]
        self,
        name: str = "",
        materials: list[str] | None = None,
        category: str | None = None,
        top_k: int = 3,
    ) -> str:
        if not name and materials:
            name = materials[0] if len(materials) == 1 else ", ".join(materials)
        if not name:
            return "Error: pass the material name via 'name' or 'materials'."
        return await self.catalog.find_similar(name, top_k=top_k)
