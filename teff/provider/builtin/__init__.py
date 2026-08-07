"""Built-in provider presets — subclasses of :class:`Provider`."""

from teff.provider.builtin.anthropic import Anthropic
from teff.provider.builtin.base import Provider
from teff.provider.builtin.deepseek import DeepSeek
from teff.provider.builtin.gemini import Gemini
from teff.provider.builtin.groq import Groq
from teff.provider.builtin.mistral import Mistral
from teff.provider.builtin.ollama import Ollama
from teff.provider.builtin.openai import OpenAI
from teff.provider.builtin.openai_compatible import OpenAICompatible
from teff.provider.builtin.openrouter import OpenRouter
from teff.provider.builtin.together import Together

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
