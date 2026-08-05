"""DeepSeek preset."""

from draf.provider.builtin.base import Provider


class DeepSeek(Provider):
    """DeepSeek's OpenAI-compatible endpoint."""

    name = "deepseek"
    base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"
