"""Gemini preset (Google's OpenAI-compatible endpoint)."""

from draf.provider.builtin.base import Provider


class Gemini(Provider):
    """Google's ``openai`` protocol endpoint."""

    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key_env = "GEMINI_API_KEY"
