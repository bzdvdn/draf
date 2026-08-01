"""Web fetch tool — download a URL and extract its text content."""

from draf.tool.tool import Tool


class WebFetchTool(Tool):
    """Fetch a URL and return the page's text content.

    Uses ``httpx`` (a core dependency) for the request and
    ``beautifulsoup4`` (from ``draf[tools]``) to strip markup.

    Args:
        config: Optional dict with ``timeout`` (seconds) and ``user_agent``.
        timeout: Request timeout in seconds (default 15).
        user_agent: User-Agent header sent with the request (default "draf").
    """

    name = "fetch_url"
    description = "Fetch a URL and extract its text content"

    def __init__(
        self,
        config: dict | None = None,
        *,
        timeout: float = 15.0,
        user_agent: str = "draf",
    ):
        if isinstance(config, dict):
            timeout = config.get("timeout", timeout)
            user_agent = config.get("user_agent", user_agent)
        self.timeout = timeout
        self.user_agent = user_agent

    async def arun(self, url: str = "", max_chars: int = 10000) -> str:  # type: ignore[override]
        if not url:
            raise ValueError("url is required")
        import httpx

        async with httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=True
        ) as client:
            response = await client.get(url, headers={"User-Agent": self.user_agent})
            response.raise_for_status()

        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            msg = "fetch_url requires 'beautifulsoup4' (pip install draf[tools])"
            raise ImportError(msg) from e

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings).strip()
        if not text:
            return "no text content found"
        return text[:max_chars]
