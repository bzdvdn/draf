"""Built-in provider presets — subclasses of :class:`Provider`."""

from draf.provider.builtin.anthropic import Anthropic
from draf.provider.builtin.base import Provider
from draf.provider.builtin.deepseek import DeepSeek
from draf.provider.builtin.gemini import Gemini
from draf.provider.builtin.groq import Groq
from draf.provider.builtin.mistral import Mistral
from draf.provider.builtin.ollama import Ollama
from draf.provider.builtin.openai import OpenAI
from draf.provider.builtin.openai_compatible import OpenAICompatible
from draf.provider.builtin.openrouter import OpenRouter
from draf.provider.builtin.together import Together

#: Built-in preset classes by name, in the order they appear in messages.
BUILTINS: dict[str, type[Provider]] = {
    "openai": OpenAI,
    "anthropic": Anthropic,
    "deepseek": DeepSeek,
    "ollama": Ollama,
    "mistral": Mistral,
    "together": Together,
    "groq": Groq,
    "openrouter": OpenRouter,
    "gemini": Gemini,
    "openai_compatible": OpenAICompatible,
}

__all__ = [
    "BUILTINS",
    "Anthropic",
    "DeepSeek",
    "Gemini",
    "Groq",
    "Mistral",
    "Ollama",
    "OpenAI",
    "OpenAICompatible",
    "OpenRouter",
    "Provider",
    "Together",
]
