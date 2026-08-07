"""OpenRouter preset."""

from teff.provider.builtin.base import Provider


class OpenRouter(Provider):
    """OpenRouter's OpenAI-compatible endpoint."""

    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"
