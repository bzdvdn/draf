"""Tests for the provider system: the ``Provider`` value object, the
``ProviderRegistry``, ``default_provider=`` plumbing, and the YAML
``providers:`` list."""

import httpx
import pytest

from draf.errors import ConfigError
from draf.provider import (
    BUILTINS,
    DEFAULT_PROVIDERS,
    PROVIDER_TYPES,
    Provider,
    ProviderRegistry,
    resolve_provider_entry,
    to_provider_registry,
)


def _mock_response(data: dict):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    return MockResponse()


@pytest.fixture
def post_bodies(monkeypatch):
    bodies = []

    async def mock_post(self, url, headers=None, json=None):
        bodies.append(json)
        if "max_tokens" in (json or {}):
            data = {
                "content": [{"type": "text", "text": "hello claude"}],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }
        elif (json or {}).get("format") == "json":
            data = {
                "message": {"role": "assistant", "content": '{"a": 1}'},
                "done": True,
            }
        else:
            data = {
                "choices": [
                    {"message": {"role": "assistant", "content": "hello openai"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        return _mock_response(data)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    return bodies


class TestProvider:
    def test_preset_types(self):
        assert DEFAULT_PROVIDERS["openai"].type == "openai_compatible"
        assert DEFAULT_PROVIDERS["anthropic"].type == "anthropic_compatible"
        assert DEFAULT_PROVIDERS["ollama"].type == "ollama"
        assert set(PROVIDER_TYPES) == {
            "openai_compatible",
            "anthropic_compatible",
            "ollama",
        }

    def test_presets_carry_names(self):
        assert BUILTINS["openai"].name == "openai"
        assert BUILTINS["anthropic"].name == "anthropic"
        assert BUILTINS["ollama"].name == "ollama"

    def test_custom_provider_with_name(self):
        p = Provider(name="my-vllm", base_url="http://vllm:8000/v1")
        assert p.name == "my-vllm"
        assert p.type == "openai_compatible"

    def test_from_mapping_keeps_only_known_fields(self):
        p = Provider.from_mapping(
            {
                "name": "x",
                "type": "anthropic_compatible",
                "base_url": "http://x",
                "bogus": 1,
            }
        )
        assert p.name == "x"
        assert p.type == "anthropic_compatible"
        assert p.base_url == "http://x"
        assert not hasattr(p, "bogus")

    def test_unknown_field_raises(self):
        with pytest.raises(TypeError, match="unknown Provider field"):
            Provider(bogus=1)

    def test_to_dict_round_trip(self):
        p = Provider(name="x", type="ollama", base_url="http://o")
        d = p.to_dict()
        assert d["name"] == "x"
        assert d["type"] == "ollama"
        assert d["base_url"] == "http://o"
        assert Provider.from_mapping(d).to_dict() == d

    def test_unknown_provider_raises(self):
        with pytest.raises(ConfigError, match="unknown provider"):
            resolve_provider_entry("not-a-real-provider")


class TestProviderRegistry:
    def test_register_and_resolve_custom(self):
        reg = ProviderRegistry()
        p = Provider(name="vllm", type="ollama", base_url="http://custom:11434")
        reg.register(p)
        assert reg.resolve("vllm") is p
        assert p.name in reg
        assert len(reg) == 1

    def test_register_returns_self_for_chaining(self):
        reg = (
            ProviderRegistry()
            .register(Provider(name="a", base_url="http://a"))
            .register(Provider(name="b", type="ollama", base_url="http://b"))
        )
        assert len(reg) == 2
        assert reg.resolve("a").base_url == "http://a"
        assert reg.resolve("b").type == "ollama"

    def test_duplicate_name_raises(self):
        reg = ProviderRegistry()
        reg.register(Provider(name="vllm", base_url="http://a"))
        with pytest.raises(ConfigError, match="already registered"):
            reg.register(Provider(name="vllm", base_url="http://b"))

    def test_register_without_name_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(ConfigError, match="without a name"):
            reg.register(Provider(base_url="http://x"))

    def test_resolve_requires_registration(self):
        reg = ProviderRegistry()
        reg.register(Provider(name="custom", base_url="http://custom"))
        with pytest.raises(ConfigError, match="unknown provider"):
            reg.resolve("openai")

    def test_from_presets_registers_builtins(self):
        reg = ProviderRegistry.from_presets("openai", "ollama")
        assert reg.resolve("openai").type == "openai_compatible"
        assert reg.resolve("ollama").type == "ollama"
        with pytest.raises(ConfigError, match="unknown preset"):
            ProviderRegistry.from_presets("nope")

    def test_resolve_unknown_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(ConfigError, match="unknown provider"):
            reg.resolve("nope")

    def test_dict_constructs_registry(self):
        reg = ProviderRegistry({"x": Provider(base_url="http://x")})
        assert reg.resolve("x").base_url == "http://x"
        assert reg.resolve("x").name == "x"

    def test_dict_construct_name_mismatch_raises(self):
        with pytest.raises(ConfigError, match="name mismatch"):
            ProviderRegistry({"x": Provider(name="y", base_url="http://x")})

    def test_resolve_provider_entry_accepts_registry(self):
        reg = ProviderRegistry()
        p = Provider(name="ollama", type="ollama", base_url="http://custom:11434")
        reg.register(p)
        assert resolve_provider_entry("ollama", reg) is p


class TestToProviderRegistry:
    def test_none_makes_empty(self):
        reg = to_provider_registry(None)
        assert isinstance(reg, ProviderRegistry)
        assert len(reg) == 0

    def test_dict_wrapped(self):
        reg = to_provider_registry({"x": Provider(base_url="http://x")})
        assert isinstance(reg, ProviderRegistry)
        assert reg.resolve("x").base_url == "http://x"

    def test_registry_passthrough(self):
        reg = ProviderRegistry()
        assert to_provider_registry(reg) is reg


class TestHarnessProviders:
    @pytest.mark.asyncio
    async def test_custom_anthropic_provider(self, post_bodies):
        from draf.harness import Harness

        providers = {
            "claude-proxy": Provider(
                type="anthropic_compatible",
                base_url="http://proxy",
                chat_path="/v1/messages",
            )
        }
        h = Harness(
            model="m",
            provider="claude-proxy",
            api_key_env="X",
            providers=providers,
        )
        assert h.type == "anthropic_compatible"
        reply = await h.call([{"role": "user", "content": "hi"}])
        assert reply.content == "hello claude"
        assert "max_tokens" in post_bodies[0]
        assert post_bodies[0]["model"] == "m"

    @pytest.mark.asyncio
    async def test_custom_ollama_structured(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        providers = {"local": Provider(type="ollama", base_url="http://ollama.local")}
        g = Graph(
            {
                "llm": LLM(
                    model="m",
                    provider="local",
                    json_schema={"type": "object"},
                )
            },
            [],
            "llm",
        )
        r = await g.run({}, providers=providers)
        assert r["output"] == {"a": 1}
        assert post_bodies[0].get("format") == "json"

    @pytest.mark.asyncio
    async def test_run_providers_override_graph_default(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        g = Graph(
            {"llm": LLM(model="m", api_key_env="X")},
            [],
            "llm",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        providers = {
            "claude": Provider(type="anthropic_compatible", base_url="http://proxy")
        }
        r = await g.run({}, default_provider="claude", providers=providers)
        assert r["output"] == "hello claude"

    @pytest.mark.asyncio
    async def test_graph_providers_used_by_default(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        providers = {
            "claude": Provider(type="anthropic_compatible", base_url="http://proxy")
        }
        g = Graph(
            {"llm": LLM(model="m", provider="claude", api_key_env="X")},
            [],
            "llm",
            providers=providers,
        )
        r = await g.run({})
        assert r["output"] == "hello claude"

    @pytest.mark.asyncio
    async def test_unknown_provider_raises_at_runtime(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        g = Graph(
            {"llm": LLM(model="m", provider="nope", api_key_env="X")},
            [],
            "llm",
        )
        with pytest.raises(ConfigError, match="not declared"):
            await g.run({})

    @pytest.mark.asyncio
    async def test_no_provider_raises_without_default(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        g = Graph({"llm": LLM(model="m", api_key_env="X")}, [], "llm")
        with pytest.raises(ConfigError, match="no provider configured"):
            await g.run({})

    @pytest.mark.asyncio
    async def test_default_provider_not_declared_raises(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        g = Graph(
            {"llm": LLM(model="m", api_key_env="X")},
            [],
            "llm",
            default_provider="nope",
        )
        with pytest.raises(ConfigError, match="default_provider"):
            await g.run({}, providers=ProviderRegistry())

    def test_from_presets_unknown_name_raises(self):
        with pytest.raises(ConfigError, match="unknown preset"):
            ProviderRegistry.from_presets("not-a-preset")

    @pytest.mark.asyncio
    async def test_graph_default_provider(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        g = Graph(
            {"llm": LLM(model="m", api_key_env="X")},
            [],
            "llm",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        r = await g.run({})
        assert r["output"] == "hello openai"

    @pytest.mark.asyncio
    async def test_run_default_provider_override(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        providers = {
            "claude": Provider(type="anthropic_compatible", base_url="http://proxy")
        }
        g = Graph(
            {"llm": LLM(model="m", api_key_env="X")},
            [],
            "llm",
            default_provider="openai",
        )
        r = await g.run({}, default_provider="claude", providers=providers)
        assert r["output"] == "hello claude"

    @pytest.mark.asyncio
    async def test_harness_accepts_registry(self, post_bodies):
        from draf.harness import Harness

        reg = ProviderRegistry()
        reg.register(
            Provider(
                name="claude-proxy",
                type="anthropic_compatible",
                base_url="http://proxy",
                chat_path="/v1/messages",
            )
        )
        h = Harness(
            model="m",
            provider="claude-proxy",
            api_key_env="X",
            providers=reg,
        )
        assert h.type == "anthropic_compatible"
        reply = await h.call([{"role": "user", "content": "hi"}])
        assert reply.content == "hello claude"

    @pytest.mark.asyncio
    async def test_graph_run_accepts_registry(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        reg = ProviderRegistry()
        reg.register(
            Provider(
                name="claude", type="anthropic_compatible", base_url="http://proxy"
            )
        )
        g = Graph(
            {"llm": LLM(model="m", provider="claude", api_key_env="X")}, [], "llm"
        )
        r = await g.run({}, providers=reg)
        assert r["output"] == "hello claude"

    @pytest.mark.asyncio
    async def test_graph_registry_used_by_default(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        reg = ProviderRegistry()
        reg.register(
            Provider(
                name="claude", type="anthropic_compatible", base_url="http://proxy"
            )
        )
        g = Graph(
            {"llm": LLM(model="m", provider="claude", api_key_env="X")},
            [],
            "llm",
            providers=reg,
        )
        assert g.providers is reg
        r = await g.run({})
        assert r["output"] == "hello claude"

    @pytest.mark.asyncio
    async def test_registry_unknown_provider_raises(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        reg = ProviderRegistry()
        g = Graph({"llm": LLM(model="m", provider="nope", api_key_env="X")}, [], "llm")
        with pytest.raises(ConfigError, match="not declared"):
            await g.run({}, providers=reg)

    @pytest.mark.asyncio
    async def test_flow_registry_compiles_into_graph(self, post_bodies):
        from draf.flow import Flow
        from draf.node import LLM

        reg = ProviderRegistry()
        reg.register(
            Provider(
                name="claude", type="anthropic_compatible", base_url="http://proxy"
            )
        )
        g = (
            Flow("reg", default_provider="claude", providers=reg)
            .step(LLM(model="m", api_key_env="X"))
            .compile()
        )
        assert g.providers is reg
        assert g.default_provider == "claude"
        r = await g.run({})
        assert r["output"] == "hello claude"


class TestStrictModel:
    def test_from_config_requires_model_or_default_model(self):
        from draf.harness import Harness

        with pytest.raises(ConfigError, match="no model configured"):
            Harness.from_config(
                {"provider": "openai", "api_key_env": "X"}, providers={}
            )

    def test_from_config_uses_default_model(self):
        from draf.harness import Harness

        h = Harness.from_config(
            {"provider": "openai", "api_key_env": "X"},
            default_model="gpt-default",
        )
        assert h.model == "gpt-default"

    @pytest.mark.asyncio
    async def test_graph_default_model_used_without_node_model(self, post_bodies):
        from draf.graph import Graph
        from draf.node import LLM

        g = Graph(
            {"llm": LLM(provider="openai", api_key_env="X")},
            [],
            "llm",
            providers=ProviderRegistry.from_presets("openai"),
            default_model="gpt-default",
        )
        r = await g.run({})
        assert r["output"] == "hello openai"


class TestYamlProviders:
    YAML = """\
name: p
default_provider: vllm
providers:
  - name: openai
    type: openai_compatible
  - name: vllm
    base_url: http://vllm:8000/v1
  - name: claude-proxy
    type: anthropic_compatible
    base_url: http://proxy
    chat_path: /v1/messages
steps:
  - id: s
    type: transform
    config: {action: trim}
edges: []
"""

    def test_providers_block_builds_graph_providers(self):
        from draf.yaml import from_yaml

        g = from_yaml(self.YAML)
        assert g.default_provider == "vllm"
        assert g.providers["vllm"].type == "openai_compatible"
        assert g.providers["vllm"].base_url == "http://vllm:8000/v1"
        assert g.providers["claude-proxy"].type == "anthropic_compatible"
        assert g.providers["claude-proxy"].chat_path == "/v1/messages"
        assert "openai" in g.providers

    def test_providers_block_round_trip(self):
        from draf.yaml import from_yaml, workflow_to_yaml

        g = from_yaml(self.YAML)
        out = workflow_to_yaml(g)
        g2 = from_yaml(out)
        assert g2.default_provider == "vllm"
        assert g2.providers["vllm"].type == "openai_compatible"
        assert g2.providers["claude-proxy"].type == "anthropic_compatible"
        assert g2.providers["claude-proxy"].chat_path == "/v1/messages"
        assert "default_provider:" in out
        assert "providers:" in out

    def test_mapping_unknown_key_raises_config_error(self):
        from draf.yaml import from_yaml

        bad = self.YAML.replace("base_url: http://proxy", "bogus: 1")
        with pytest.raises(ConfigError, match="unknown keys"):
            from_yaml(bad)

    def test_duplicate_name_raises(self):
        from draf.yaml import from_yaml

        bad = self.YAML.replace(
            "  - name: claude-proxy", "  - name: claude-proxy\n  - name: claude-proxy"
        )
        with pytest.raises(ConfigError, match="already registered"):
            from_yaml(bad)

    def test_preset_name_string_is_rejected(self):
        from draf.yaml import from_yaml

        bad = self.YAML.replace(
            "  - name: openai\n    type: openai_compatible\n", "  - openai\n"
        )
        with pytest.raises(ConfigError, match="not of type 'object'"):
            from_yaml(bad)

    def test_runtime_parser_rejects_bare_strings(self):
        from draf.errors import ConfigError as CE
        from draf.yaml import _providers_from_data

        with pytest.raises(CE, match="preset names are not allowed"):
            _providers_from_data({"providers": ["openai"]})

    def test_runtime_parser_rejects_non_mapping_entries(self):
        from draf.errors import ConfigError as CE
        from draf.yaml import _providers_from_data

        with pytest.raises(CE, match="must be a mapping"):
            _providers_from_data({"providers": [42]})

    def test_runtime_parser_requires_name(self):
        from draf.errors import ConfigError as CE
        from draf.yaml import _providers_from_data

        with pytest.raises(CE, match="requires a `name:`"):
            _providers_from_data({"providers": [{"type": "ollama"}]})

    def test_default_provider_must_be_declared(self):
        from draf.yaml import from_yaml

        bad = self.YAML.replace("default_provider: vllm", "default_provider: nope")
        with pytest.raises(ConfigError, match="default_provider"):
            from_yaml(bad)

    def test_node_provider_must_be_declared(self):
        from draf.yaml import from_yaml

        yaml_with_node_provider = """\
name: p
providers:
  - name: openai
    type: openai_compatible
steps:
  - id: s
    type: llm_chat
    config:
      model: gpt-4o
      provider: undeclared
edges: []
"""
        with pytest.raises(ConfigError, match="undeclared"):
            from_yaml(yaml_with_node_provider)

    def test_builtin_node_provider_is_allowed(self):
        from draf.yaml import from_yaml

        yaml_ok = """\
name: p
default_provider: openai
providers:
  - name: openai
    type: openai_compatible
steps:
  - id: s
    type: llm_chat
    config:
      model: gpt-4o
      provider: openai
edges: []
"""
        g = from_yaml(yaml_ok)
        assert g.default_provider == "openai"


def test_provider_submodules_import_cleanly():
    """Re-execute the provider submodules whose module-level code runs before
    coverage starts (they are imported by the ``draf.testing`` pytest plugin).
    Running the reload last avoids disturbing class identity for earlier tests."""
    import importlib

    for mod in (
        "draf.provider.base",
        "draf.provider.registry",
        "draf.provider.resolve",
        "draf.provider.providers",
    ):
        importlib.reload(importlib.import_module(mod))
    assert ProviderRegistry.from_presets("ollama").resolve("ollama").type == "ollama"
