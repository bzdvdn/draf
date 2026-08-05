"""Generic OpenAI-compatible provider (vLLM, LM Studio, Azure, ...)."""

from draf.provider.builtin.base import Provider


class OpenAICompatible(Provider):
    """Any ``/chat/completions`` endpoint; set ``base_url`` yourself."""

    name = "openai_compatible"
    base_url = ""
    api_key_env = "OPENAI_API_KEY"
