"""Global per-provider concurrency guards.

Shared across :class:`~draf.harness.Harness` instances so parallel branches
(each with its own harness) throttle model traffic together instead of
blowing past provider rate limits.
"""

import asyncio

#: Global per-provider semaphores keyed by provider key.
_PROVIDER_SEMAPHORES: dict[str, asyncio.Semaphore] = {}

#: Capacity of each provider's semaphore (mirrors ``_value`` without
#: touching the private attribute).
_PROVIDER_LIMITS: dict[str, int] = {}

#: Number of goroutines currently holding a provider's semaphore.  Tracks
#: idleness without reading the private ``Semaphore._value``.
_PROVIDER_ACTIVE: dict[str, int] = {}

#: Providers with an explicit global cap (authoritative over ``max_parallel``).
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
        _PROVIDER_LIMITS.pop(provider, None)
        _PROVIDER_ACTIVE.pop(provider, None)
    else:
        _EXPLICIT_LIMITS[provider] = limit
        _PROVIDER_SEMAPHORES[provider] = asyncio.Semaphore(limit)
        _PROVIDER_LIMITS[provider] = limit
        _PROVIDER_ACTIVE[provider] = 0


def provider_concurrency(provider: str) -> int | None:
    """Return the current global concurrency limit for *provider* (if any).

    Returns the active semaphore's capacity (explicit or auto-grown via
    ``max_parallel``), or ``None`` when the provider has no semaphore.
    """
    return _PROVIDER_LIMITS.get(provider.lower())
