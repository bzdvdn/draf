"""Backward-compatible alias for :mod:`teff.provider.providers`.

Provider machinery now lives in the :mod:`teff.provider` package; this
module only re-exports it so ``from teff.harness.providers import ...``
keeps working.
"""

from teff.provider.providers import (  # noqa: F401
    _EXPLICIT_LIMITS,
    _JSON_TYPE_MAP,
    _PRESET_NAMES,
    _PROVIDER_SEMAPHORES,
    DEFAULT_PROVIDERS,
    PROVIDER_DEFAULTS,
    PROVIDER_TYPES,
    Provider,
    ProviderRegistry,
    _preset,
    _provider_type,
    provider_concurrency,
    resolve_provider,
    resolve_provider_entry,
    set_provider_concurrency,
)
