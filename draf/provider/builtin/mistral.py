"""Mistral preset."""

from draf.provider.builtin.base import Provider


class Mistral(Provider):
    """Mistral's OpenAI-compatible endpoint."""

    name = "mistral"
    base_url = "https://api.mistral.ai/v1"
    api_key_env = "MISTRAL_API_KEY"