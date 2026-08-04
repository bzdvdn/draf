"""OpenAI preset."""

from draf.provider.base import Provider


class OpenAI(Provider):
    """OpenAI chat-completions endpoint."""

    name = "openai"
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"
