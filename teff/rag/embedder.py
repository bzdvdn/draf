"""Embedding service using provider APIs."""

import os
from dataclasses import dataclass

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
        api_key_env: Env var holding the API key; defaults to the
            per-provider env var.
    """

    provider: str = "openai"
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None

    def __post_init__(self):
        default_url, default_env, default_model = _EMBEDDER_DEFAULTS.get(
            self.provider, _EMBEDDER_DEFAULTS["openai"]
        )
        self._base_url = self.base_url or os.environ.get(
            f"{self.provider.upper()}_BASE_URL", default_url
        )
        env_key = default_env if self.api_key_env is None else self.api_key_env
        self._api_key = os.environ.get(
            f"{self.provider.upper()}_API_KEY",
            os.environ.get(env_key, ""),
        )
        if env_key and not self._api_key:
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


def embedder_from_config(
    cfg: dict, *, providers=None, default_provider: str | None = None
) -> Embedder:
    """Build an :class:`Embedder` from an ``embedder:`` config dict.

    Explicit ``provider`` / ``model`` / ``base_url`` / ``api_key_env`` in
    the config win.  When a piece is missing, values from the matching
    :class:`~teff.provider.Provider` in *providers* (a registry) are used.

    The embeddings endpoint is OpenAI-compatible (``<base>/embeddings``), so
    an inherited ``base_url`` gets a ``/v1`` suffix appended unless it
    already has one — e.g. an ``ollama`` provider whose ``base_url`` points
    at its native ``/api/chat`` host becomes ``http://localhost:11434/v1``.
    ``api_key_env`` is inherited from any provider that declares one, so a
    custom key env var used for chat also applies to embeddings.
    """
    emb = cfg.get("embedder") or {}
    provider = emb.get("provider") or default_provider or "ollama"
    base_url = emb.get("base_url")
    api_key_env = emb.get("api_key_env")
    if providers is not None and emb.get("base_url") is None:
        try:
            p = providers.resolve(provider)
        except Exception:
            p = None
        if p is not None:
            if p.base_url:
                base_url = _with_embeddings_path(p.base_url)
            if p.api_key_env:
                api_key_env = p.api_key_env
            elif not api_key_env:
                # A registry-declared provider with no key env var is
                # self-configuring (like ollama): don't demand a key.
                api_key_env = ""
    return Embedder(
        provider=provider,
        model=emb.get("model") or "",
        base_url=base_url,
        api_key_env=api_key_env,
    )


def _with_embeddings_path(url: str) -> str:
    """Ensure *url* points at an OpenAI-compatible embeddings root (``/v1``)."""
    stripped = url.rstrip("/")
    return stripped if stripped.endswith("/v1") else stripped + "/v1"
