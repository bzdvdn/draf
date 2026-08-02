"""RAG tools backed by the document catalog (``rag`` variant).

The tool names here must match the allowlist threaded into the writer agent
by ``src.graphs.build`` (``_RAG_TOOLS``).  Swap the catalog for a real
service and only this module changes.
"""

from draf.tool.tool import Tool


class SearchCatalog(Tool):
    name = "search_catalog"
    description = (
        "Search the document catalog by query; returns ranked snippets. "
        "Useful when the user asks about facts covered by indexed documents."
    )

    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

    async def arun(self, query: str, top_k: int | None = None) -> str:  # type: ignore[override]
        return await self.catalog.search(query, top_k=top_k)


class FindSimilar(Tool):
    name = "find_similar"
    description = "Find documents similar to the given text or description."

    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

    async def arun(self, text: str, top_k: int = 3) -> str:  # type: ignore[override]
        return await self.catalog.find_similar(text, top_k=top_k)
