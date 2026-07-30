"""Embedding service using provider APIs."""

from dataclasses import dataclass
import os

import httpx


@dataclass
class Embedder:
    """Convert text to vector embeddings using a provider API.

    Attributes:
        provider: Provider name (default: ``openai``).
        model: Embedding model name.
        base_url: Optional custom API base URL.
    """

    provider: str = "openai"
    model: str = "text-embedding-ada-002"
    base_url: str | None = None

    def __post_init__(self):
        self._base_url = self.base_url or os.environ.get(
            f"{self.provider.upper()}_BASE_URL",
            "https://api.openai.com/v1",
        )
        self._api_key = os.environ.get(
            f"{self.provider.upper()}_API_KEY",
            os.environ.get("OPENAI_API_KEY", ""),
        )
        if not self._api_key:
            raise ValueError(f"API key not found for provider {self.provider}")

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_many([text])
        return results[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
