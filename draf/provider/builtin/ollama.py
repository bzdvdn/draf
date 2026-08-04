"""Ollama preset (local)."""

from draf.provider.builtin.base import Provider


class Ollama(Provider):
    """Ollama's local ``/api/chat`` endpoint."""

    name = "ollama"
    type = "ollama"
    base_url = "http://localhost:11434"
    chat_path = "/api/chat"
    api_key_env = ""
    auth_header = ""
    auth_prefix = ""