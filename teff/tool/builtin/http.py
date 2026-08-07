"""HTTP tool — send arbitrary HTTP requests to APIs."""

import json

from teff.tool.tool import Tool


class HttpRequestTool(Tool):
    """Send an HTTP request to an API endpoint.

    Unlike :class:`~teff.tool.builtin.web_fetch.WebFetchTool`, this tool
    exposes the full request surface: method, headers, and body, and
    returns the raw response (status, headers, text).

    Args:
        config: Optional dict with ``timeout`` (seconds, default 30).
    """

    name = "http_request"
    description = "Send an HTTP request (GET/POST/PUT/DELETE) and return the response"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.timeout = cfg.get("timeout", 30.0)

    async def arun(  # type: ignore[override]
        self,
        url: str = "",
        method: str = "GET",
        headers: str = "",
        body: str = "",
        max_chars: int = 20000,
    ) -> str:
        if not url:
            raise ValueError("url is required")
        import httpx

        request_headers: dict | None = None
        if headers:
            try:
                request_headers = json.loads(headers)
            except json.JSONDecodeError as e:
                raise ValueError("headers must be a JSON object string") from e

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method.upper(),
                url,
                headers=request_headers,
                content=body.encode("utf-8") if body else None,
            )

        response_headers = "\n".join(f"{k}: {v}" for k, v in response.headers.items())
        text = response.text[:max_chars]
        return f"HTTP {response.status_code}\n{response_headers}\n\n{text}"
