"""Together preset."""

from draf.provider.base import Provider


class Together(Provider):
    """Together.ai's OpenAI-compatible endpoint."""

    name = "together"
    base_url = "https://api.together.xyz/v1"
    api_key_env = "TOGETHER_API_KEY"
