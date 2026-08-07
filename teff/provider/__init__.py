"""Providers — wire protocols, presets, and the registry.

A :class:`Provider` is a named endpoint: its ``type`` selects the wire
protocol (``openai_compatible`` / ``anthropic_compatible`` / ``ollama``)
and ``base_url``/``chat_path``/auth keys point at the endpoint.  Providers
are declared in a workflow's ``providers:`` block, registered once in a
:class:`ProviderRegistry`, or picked from the built-in presets in
:data:`DEFAULT_PROVIDERS`.
"""

from teff.provider.builtin.base import PROVIDER_FIELDS, Provider
from teff.provider.concurrency import (
    provider_concurrency,
    set_provider_concurrency,
)
from teff.provider.registry import BUILTINS, DEFAULT_PROVIDERS, ProviderRegistry
from teff.provider.resolve import (
    PROVIDER_TYPES,
    resolve_provider,
    resolve_provider_entry,
    to_provider_registry,
    validate_provider_refs,
)

#: Backward-compatible ``{name: {endpoint fields}}`` preset defaults.
PROVIDER_DEFAULTS: dict = {name: cls().to_dict() for name, cls in BUILTINS.items()}

__all__ = [
    "BUILTINS",
    "DEFAULT_PROVIDERS",
    "PROVIDER_DEFAULTS",
    "PROVIDER_FIELDS",
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
