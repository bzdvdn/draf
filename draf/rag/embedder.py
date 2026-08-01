"""Embedding service using provider APIs."""

from dataclasses import dataclass
import os

import httpx

_EMBEDDER_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "text-embedding-ada-002"),
    "ollama": ("http://localhost:11434/v1", "", "nomic-embed-text"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", "mistral-embed"),
    "voyage": ("https://api.voyageai.com/v1", "VOYAGE_API_KEY", "voyage-3"),
    "jina": ("https://api.jina.ai/v1", "JINA_API_KEY", "jina-embeddings-v3"),
    "together": (
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
        "togethercomputer/m2-bert-80M-8k-retrieval",
    ),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "nomic-embed-text-v1.5"),
}


@dataclass
class Embedder:
    """Convert text to vector embeddings using a provider API.

    Attributes:
        provider: Provider name (``openai``, ``ollama``, ``mistral``,
            ``voyage``, ``jina``, ``together``, ``groq``, ...). Any
            OpenAI-compatible ``/v1/embeddings`` endpoint works.
        model: Embedding model name; defaults to a per-provider model.
        base_url: Optional custom API base URL.
    """

    provider: str = "openai"
    model: str = ""
    base_url: str | None = None

    def __post_init__(self):
        default_url, default_env, default_model = _EMBEDDER_DEFAULTS.get(
            self.provider, _EMBEDDER_DEFAULTS["openai"]
        )
        self._base_url = self.base_url or os.environ.get(
            f"{self.provider.upper()}_BASE_URL", default_url
        )
        self._api_key = os.environ.get(
            f"{self.provider.upper()}_API_KEY",
            os.environ.get(default_env, ""),
        )
        if default_env and not self._api_key:
            raise ValueError(f"API key not found for provider {self.provider}")
        if not self.model:
            self.model = default_model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_many([text])
        return results[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json={
                    "model": self.model,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()
            return [
                item["embedding"]
                for item in sorted(data["data"], key=lambda x: x["index"])
            ]
