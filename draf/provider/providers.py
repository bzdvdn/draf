"""Backward-compatible alias for the :mod:`draf.provider` package.

Provider machinery now lives in focused submodules (``base``, ``registry``,
``resolve``, ``concurrency``, and one module per built-in preset); this
module only re-exports it so ``from draf.provider.providers import ...``
keeps working.
"""

from draf.provider.base import PROVIDER_FIELDS, Provider  # noqa: F401
from draf.provider.concurrency import (  # noqa: F401
    _EXPLICIT_LIMITS,
    _PROVIDER_SEMAPHORES,
    provider_concurrency,
    set_provider_concurrency,
)
from draf.provider.registry import (  # noqa: F401
    BUILTINS,
    DEFAULT_PROVIDERS,
    ProviderRegistry,
)
from draf.provider.resolve import (  # noqa: F401
    PROVIDER_TYPES,
    resolve_provider,
    resolve_provider_entry,
    to_provider_registry,
    validate_provider_refs,
)

#: JSON type map used when building tool/parameter schemas.
_JSON_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

#: Built-in preset names in display order.
_PRESET_NAMES = tuple(BUILTINS)

#: Backward-compatible ``{name: {endpoint fields}}`` preset defaults.
PROVIDER_DEFAULTS: dict = {name: cls().to_dict() for name, cls in BUILTINS.items()}


def _provider_type(name: str) -> str:
    """Wire-protocol ``type`` for a built-in preset name."""
    cls = BUILTINS.get(name)
    return cls.type if cls is not None else "openai_compatible"


def _preset(name: str) -> dict:
    """Endpoint defaults for a preset name (unknown → OpenAI shape)."""
    cls = BUILTINS.get(name, BUILTINS["openai_compatible"])
    return cls().to_dict()


__all__ = [
    "BUILTINS",
    "DEFAULT_PROVIDERS",
    "PROVIDER_DEFAULTS",
    "PROVIDER_TYPES",
    "Provider",
    "ProviderRegistry",
    "provider_concurrency",
    "resolve_provider",
    "resolve_provider_entry",
    "set_provider_concurrency",
    "to_provider_registry",
    "validate_provider_refs",
]
