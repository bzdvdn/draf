"""The :class:`ProviderRegistry` and the built-in preset catalogue.

Register :class:`~teff.provider.builtin.base.Provider` instances once and
reference them by ``name`` anywhere a provider key is expected (``provider=``
on nodes, ``default_provider=`` on the graph, ``providers=`` at run time,
``providers:`` in YAML).  The registry starts empty and is the single source
of truth: a provider is only usable once it has been explicitly registered
(built-in presets are registered as instances, e.g.
``providers=ProviderRegistry.from_presets("ollama")``).
"""

from teff.errors import ConfigError
from teff.provider.builtin import BUILTINS, Provider

#: Built-in providers as fresh instances (backward-compatible dict).
DEFAULT_PROVIDERS: dict[str, Provider] = {name: cls() for name, cls in BUILTINS.items()}


class ProviderRegistry:
    """A named collection of providers.

    Register providers once and reference them by name anywhere a provider
    key is expected.  The registry starts empty and is authoritative: an
    unregistered name raises :class:`ConfigError` rather than silently
    loading a built-in preset — every provider a graph uses must be
    declared here.

    Because the canonical input is a dict ``{name: Provider}``, the same
    value can also be built from one and consumed with :func:`dict`-style
    lookups, so ``graph.run(state, providers={...})`` keeps working.

    Example::

        from teff import Provider, ProviderRegistry

        reg = ProviderRegistry()
        reg.register(Provider(name="vllm", base_url="http://vllm:8000/v1"))
        reg.register(AnthropicCompatibleProvider(name="claude-proxy", base_url="http://proxy"))

        graph = Graph(
            {"llm": LLM(model="m", provider="claude-proxy")}, [], "llm",
            providers=reg,
            default_provider="claude-proxy",
        )
        await graph.run({})

    Registering a name that is already registered (including by a previous
    call) raises :class:`ConfigError` — names are unique.
    """

    def __init__(self, providers: "dict[str, Provider] | None" = None):
        self._entries: dict[str, Provider] = {}
        if providers:
            for name, provider in providers.items():
                if not provider.name:
                    provider.name = name
                elif provider.name != name:
                    raise ConfigError(
                        f"provider name mismatch: registered as {name!r} "
                        f"but Provider.name is {provider.name!r}"
                    )
                self.register(provider)

    def register(self, provider: Provider) -> "ProviderRegistry":
        """Add *provider* under ``provider.name``; returns self for chaining.

        Raises:
            ConfigError: If ``provider.name`` is empty or already registered.
        """
        if not provider.name:
            raise ConfigError("cannot register a Provider without a name")
        if provider.name in self._entries:
            raise ConfigError(f"provider {provider.name!r} is already registered")
        self._entries[provider.name] = provider
        return self

    def resolve(self, name: str) -> Provider:
        """Resolve *name* to a registered :class:`Provider`.

        Only explicitly registered providers are usable.  Unknown names
        raise :class:`ConfigError`.
        """
        if name in self._entries:
            return self._entries[name]
        raise ConfigError(
            f"unknown provider: {name!r} — declare it by registering it in "
            "the ProviderRegistry, a `providers=`/`providers:` block, or "
            f"naming a built-in preset ({', '.join(BUILTINS)})"
        )

    @classmethod
    def from_presets(cls, *names: str) -> "ProviderRegistry":
        """Build a registry holding the named built-in presets.

        Convenience for declaring built-ins explicitly::

            reg = ProviderRegistry.from_presets("openai", "ollama")

        An unknown preset name raises :class:`ConfigError`.
        """
        reg = cls()
        for name in names:
            preset = BUILTINS.get(name)
            if preset is None:
                raise ConfigError(
                    f"unknown preset: {name!r} — pick from {', '.join(BUILTINS)}"
                )
            reg.register(preset())
        return reg

    def items(self) -> list[tuple[str, Provider]]:
        """Registered ``(name, provider)`` pairs (custom entries only)."""
        return list(self._entries.items())

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> Provider:
        return self._entries[name]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
