"""Web search tool — DuckDuckGo search without API key."""

import httpx

from draf.tool.tool import Tool


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo (no API key required)."""

    name = "web_search"
    description = "Search the web"

    def __init__(self, provider: str = "duckduckgo"):
        self.provider = provider

    async def arun(self, query: str = "", num_results: int = 5) -> str:  # type: ignore[override]
        if self.provider == "duckduckgo":
            return await self._duckduckgo(query, num_results)
        msg = f"unknown web search provider: {self.provider}"
        raise ValueError(msg)

    async def _duckduckgo(self, query: str, num_results: int) -> str:
        url = "https://lite.duckduckgo.com/lite/"
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, data={"q": query})
            response.raise_for_status()

        import html
        text = html.unescape(response.text)
        lines = []
        count = 0
        in_link = False
        for part in text.split("<"):
            if part.startswith("a "):
                in_link = True
                continue
            if part.startswith("/a"):
                in_link = False
                continue
            if in_link:
                if ">" in part:
                    content = part.split(">", 1)[1]
                    content = content.strip()
                    if content and not content.startswith("<"):
                        lines.append(content)
                        count += 1
                        if count >= num_results:
                            break

        return "\n".join(lines) if lines else "no results"
