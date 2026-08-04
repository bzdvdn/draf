"""Provider resolution helpers shared by the harness and graph layers."""

from draf.errors import ConfigError
from draf.provider.builtin.base import Provider
from draf.provider.registry import BUILTINS, DEFAULT_PROVIDERS, ProviderRegistry

#: Wire-protocol families a :class:`Provider` can speak.
PROVIDER_TYPES = ("openai_compatible", "anthropic_compatible", "ollama")


def to_provider_registry(
    providers: "ProviderRegistry | dict[str, Provider] | None",
) -> ProviderRegistry:
    """Normalize *providers* into a :class:`ProviderRegistry`.

    Accepts an existing :class:`ProviderRegistry`, a ``{name: Provider}``
    dict, or ``None`` (empty registry).  There is no string shorthand —
    every provider must be an explicit instance, so ``graph.providers``
    truthfully reflects what is configured.
    """
    if isinstance(providers, ProviderRegistry):
        return providers
    if providers is None:
        return ProviderRegistry()
    return ProviderRegistry(providers)


def validate_provider_refs(
    providers: ProviderRegistry,
    default_provider: str | None = None,
    nodes: "dict[str, Provider] | dict | None" = None,
) -> None:
    """Enforce that every provider reference is declared in *providers*.

    *default_provider* and each node's ``config.provider`` must name a
    provider registered in *providers* — there is no implicit built-in
    fallback.  Raises :class:`ConfigError` on the first undeclared name.
    """
    valid = set(providers)
    if default_provider and default_provider not in valid:
        raise ConfigError(
            f"default_provider {default_provider!r} is not declared in "
            "`providers=` / `providers:`"
        )
    for nid, node in (nodes or {}).items():
        cfg = getattr(node, "config", None) or {}
        prov = cfg.get("provider")
        if prov and prov not in valid:
            raise ConfigError(
                f"node {nid!r}: provider {prov!r} is not declared in "
                "`providers=` / `providers:`"
            )


def resolve_provider_entry(
    provider_key: str,
    providers: "dict[str, Provider] | ProviderRegistry | None" = None,
) -> Provider:
    """Resolve the effective :class:`Provider` for *provider_key*.

    When *providers* is a :class:`ProviderRegistry` or dict it is
    authoritative — *provider_key* must be declared in it.  With ``None``
    (a bare, standalone ``Harness``) a built-in preset is used.  Unknown
    names raise a :class:`ConfigError` — there is no silent fallback to the
    OpenAI shape, so typos surface early instead of silently routing to the
    wrong wire protocol.

    Raises:
        ConfigError: When *provider_key* is neither declared in *providers*
            nor (with ``providers=None``) a built-in preset name.
    """
    if isinstance(providers, ProviderRegistry):
        return providers.resolve(provider_key)
    if providers and provider_key in providers:
        return providers[provider_key]
    if providers is None:
        preset = BUILTINS.get(provider_key)
        if preset is not None:
            return preset()
    raise ConfigError(
        f"unknown provider: {provider_key!r} — declare it in the `providers=` "
        f"map / `providers:` block, or name a built-in preset "
        f"({', '.join(BUILTINS)})"
    )


def resolve_provider(
    provider: str | None = None, default_provider: str | None = None
) -> str:
    """Resolve a provider key from an explicit value or a default name.

    The explicit *provider* (node-level) wins; otherwise *default_provider*
    (the graph-level default) is used.  Model-name auto-detection was removed
    — a provider must be stated explicitly.

    Raises:
        ConfigError: When neither a provider nor a default is configured.
    """
    p = provider or default_provider
    if not p:
        raise ConfigError(
            "no provider configured: set `provider=` on the node, pass "
            "`default_provider=` to the graph, or declare a top-level "
            "`default_provider:` in the workflow"
        )
    return p.lower()


__all__ = [
    "DEFAULT_PROVIDERS",
    "PROVIDER_TYPES",
    "resolve_provider",
    "resolve_provider_entry",
    "to_provider_registry",
    "validate_provider_refs",
]
