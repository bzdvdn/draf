"""Anthropic preset (responses normalised to a shared message shape)."""

from teff.provider.builtin.base import Provider


class Anthropic(Provider):
    """Anthropic ``/messages`` endpoint."""

    name = "anthropic"
    type = "anthropic_compatible"
    base_url = "https://api.anthropic.com/v1"
    chat_path = "/messages"
    api_key_env = "ANTHROPIC_API_KEY"
    auth_header = "x-api-key"
    auth_prefix = ""
