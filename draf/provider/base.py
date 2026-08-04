"""The :class:`Provider` value object.

A provider is a named model endpoint: how to speak to it (``type`` selects
the wire protocol) and where it lives (``base_url`` / ``chat_path`` / auth
keys).  Built-in presets are subclasses that set the defaults; custom
providers are plain instances declared in a workflow's ``providers:`` block
or passed to ``graph.run(providers=...)``.
"""

PROVIDER_FIELDS = (
    "name",
    "type",
    "base_url",
    "chat_path",
    "api_key_env",
    "auth_header",
    "auth_prefix",
    "timeout",
)


class Provider:
    """A named model endpoint: wire protocol + endpoint data.

    ``name`` is the registry key used by ``provider=`` references.
    ``type`` is the protocol discriminator — ``openai_compatible`` /
    ``anthropic_compatible`` / ``ollama`` — and decides the request body,
    streaming chunk parsing, and response extraction held by
    :class:`~draf.harness.Harness`.

    Built-in presets subclass this and set ``name`` (and the other fields)
    once; a custom provider is a plain instance.  Fields may be overridden
    at construction:

        Provider(name="my-vllm", type="openai_compatible", base_url="http://vllm:8000/v1")

    ``type`` is deliberately a distinct concept from ``name``: the name is
    just a key and never carries protocol meaning.
    """

    name: str = ""
    type: str = "openai_compatible"
    base_url: str = ""
    chat_path: str = "/chat/completions"
    api_key_env: str = ""
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    timeout: float = 120.0

    def __init__(self, **overrides):
        unknown = set(overrides) - set(PROVIDER_FIELDS)
        if unknown:
            raise TypeError(f"unknown Provider field(s): {', '.join(sorted(unknown))}")
        for field in PROVIDER_FIELDS:
            if field in overrides:
                setattr(self, field, overrides[field])

    @classmethod
    def from_mapping(cls, cfg: dict) -> "Provider":
        """Build from a config dict, keeping only known fields."""
        return cls(**{f: cfg[f] for f in PROVIDER_FIELDS if f in cfg})

    def to_dict(self) -> dict:
        """All provider fields as a plain dict (for YAML serialisation)."""
        return {f: getattr(self, f) for f in PROVIDER_FIELDS}

    def __repr__(self) -> str:
        shown = ", ".join(
            f"{f}={getattr(self, f)!r}" for f in PROVIDER_FIELDS if getattr(self, f)
        )
        return f"{type(self).__name__}({shown})"
