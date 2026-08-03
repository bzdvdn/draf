"""Provider presets, concurrency guards, and provider resolution."""

from __future__ import annotations

import asyncio

PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "OPENAI_API_KEY",
        "chat_path": "/chat/completions",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "api_key_env": "ANTHROPIC_API_KEY",
        "chat_path": "/messages",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "DEEPSEEK_API_KEY",
        "chat_path": "/chat/completions",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "auth_header": "",
        "auth_prefix": "",
        "api_key_env": "",
        "chat_path": "/api/chat",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "MISTRAL_API_KEY",
        "chat_path": "/chat/completions",
    },
    # OpenAI-compatible endpoints.
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "TOGETHER_API_KEY",
        "chat_path": "/chat/completions",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "GROQ_API_KEY",
        "chat_path": "/chat/completions",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "OPENROUTER_API_KEY",
        "chat_path": "/chat/completions",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "GEMINI_API_KEY",
        "chat_path": "/chat/completions",
    },
    # Custom OpenAI-compatible endpoint (e.g. vLLM, LM Studio, Azure).
    "openai_compatible": {
        "base_url": "",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "OPENAI_API_KEY",
        "chat_path": "/chat/completions",
    },
}

_JSON_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

# Global per-provider concurrency guards.  Shared across harness
# instances so parallel branches (each with its own Harness) throttle
# model traffic together instead of blowing past provider rate limits.
_PROVIDER_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
# Providers with an explicit global cap (authoritative over max_parallel).
_EXPLICIT_LIMITS: dict[str, int] = {}


def set_provider_concurrency(provider: str, limit: int) -> None:
    """Globally cap concurrent model calls for *provider*.

    Overrides any per-harness ``max_parallel`` for that provider.
    Pass ``limit <= 0`` to remove the cap.
    """
    provider = provider.lower()
    if limit <= 0:
        _EXPLICIT_LIMITS.pop(provider, None)
        _PROVIDER_SEMAPHORES.pop(provider, None)
    else:
        _EXPLICIT_LIMITS[provider] = limit
        _PROVIDER_SEMAPHORES[provider] = asyncio.Semaphore(limit)


def provider_concurrency(provider: str) -> int | None:
    """Return the current global concurrency limit for *provider* (if any)."""
    sem = _PROVIDER_SEMAPHORES.get(provider.lower())
    return sem._value if sem is not None else None


def resolve_provider(
    model: str, provider: str | None = None, default_provider: str | None = None
) -> str:
    """Resolve a provider key from an explicit value or the model name."""
    p = provider or default_provider
    if p:
        return p.lower()
    detected = model.split("-")[0].split("/")[0]
    return detected.lower()
