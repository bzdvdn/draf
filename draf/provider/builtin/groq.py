"""Groq preset."""

from draf.provider.builtin.base import Provider


class Groq(Provider):
    """Groq's OpenAI-compatible endpoint."""

    name = "groq"
    base_url = "https://api.groq.com/openai/v1"
    api_key_env = "GROQ_API_KEY"
