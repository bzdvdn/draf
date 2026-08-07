"""Tests for release harness features: LLM cache, Anthropic format,
concurrency throttle, and trace cost reporting."""

import asyncio

import httpx
import pytest


def _mock_response(data: dict):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    return MockResponse()


@pytest.fixture
def patch_post(monkeypatch):
    calls = {"n": 0, "active": 0, "max_active": 0, "bodies": []}

    async def mock_post(self, url, headers=None, json=None):
        calls["n"] += 1
        calls["active"] += 1
        calls["max_active"] = max(calls["max_active"], calls["active"])
        calls["bodies"].append(json)
        await asyncio.sleep(0.001)
        calls["active"] -= 1
        return _mock_response(
            {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    return calls


class TestLLMCache:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_network(self, patch_post):
        from teff.harness import Harness

        h = Harness(model="gpt-4", provider="openai", api_key_env="X", cache=True)
        h._headers = {}
        msgs = [{"role": "user", "content": "hi"}]
        first = await h.call(msgs)
        second = await h.call(msgs)
        assert patch_post["n"] == 1
        assert first.cached is False
        assert second.cached is True
        assert second.content == "hi"

    @pytest.mark.asyncio
    async def test_cache_distinguishes_messages(self, patch_post):
        from teff.harness import Harness

        h = Harness(model="gpt-4", provider="openai", api_key_env="X", cache=True)
        h._headers = {}
        await h.call([{"role": "user", "content": "a"}])
        await h.call([{"role": "user", "content": "b"}])
        assert patch_post["n"] == 2

    @pytest.mark.asyncio
    async def test_shared_mapping_cache(self, patch_post):
        from teff.harness import Harness

        shared: dict = {}
        h1 = Harness(model="gpt-4", provider="openai", api_key_env="X", cache=shared)
        h2 = Harness(model="gpt-4", provider="openai", api_key_env="X", cache=shared)
        h1._headers = {}
        h2._headers = {}
        msgs = [{"role": "user", "content": "same"}]
        await h1.call(msgs)
        reply = await h2.call(msgs)
        assert reply.cached is True
        assert patch_post["n"] == 1

    @pytest.mark.asyncio
    async def test_no_cache_by_default(self, patch_post):
        from teff.harness import Harness

        h = Harness(model="gpt-4", provider="openai", api_key_env="X")
        h._headers = {}
        msgs = [{"role": "user", "content": "hi"}]
        await h.call(msgs)
        await h.call(msgs)
        assert patch_post["n"] == 2

    def test_from_config_cache_true(self):
        from teff.harness import Harness

        h = Harness.from_config({"model": "gpt-4", "cache": True, "provider": "openai"})
        assert h._cache is not None


class TestAnthropic:
    def test_anthropic_body_builds_blocks(self):
        from teff.harness import Harness

        h = Harness(model="claude-3-5-sonnet", provider="anthropic", api_key_env="X")
        h._headers = {}
        body = h._body(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": '{"x": 1}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "1"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "calc",
                        "description": "d",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        assert body["system"] == "Be helpful."
        assert body["max_tokens"] == 1024
        assert body["messages"][1] == {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "c1", "name": "calc", "input": {"x": 1}},
            ],
        }
        assert body["messages"][2]["content"][0]["type"] == "tool_result"
        assert body["tools"][0]["name"] == "calc"
        assert body["tools"][0]["input_schema"] == {"type": "object"}

    @pytest.mark.asyncio
    async def test_anthropic_response_parsing(self, monkeypatch):
        from teff.harness import Harness

        calls = {"n": 0}

        async def mock_post(self, url, headers=None, json=None):
            calls["n"] += 1
            return _mock_response(
                {
                    "content": [
                        {"type": "text", "text": "Sure!"},
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "calculator",
                            "input": {"expression": "2+2"},
                        },
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 5},
                }
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
        h = Harness(model="claude-3-5-sonnet", provider="anthropic", api_key_env="X")
        h._headers = {}
        reply = await h.call([{"role": "user", "content": "2+2"}])
        assert reply.content == "Sure!"
        assert reply.message["tool_calls"][0]["function"]["name"] == "calculator"
        assert reply.usage == {"prompt": 12, "completion": 5}

    def test_anthropic_stream_token(self):
        from teff.harness import Harness

        h = Harness(model="claude-3-5-sonnet", provider="anthropic", api_key_env="X")
        assert (
            h._stream_token(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "hi"},
                }
            )
            == "hi"
        )
        assert (
            h._stream_token(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": ""},
                }
            )
            == ""
        )
        h2 = Harness(model="gpt-4", provider="openai", api_key_env="X")
        assert h2._stream_token({"choices": [{"delta": {"content": "yo"}}]}) == "yo"

    def test_openai_compatible_presets(self):
        from teff.harness import PROVIDER_DEFAULTS

        for provider in (
            "together",
            "groq",
            "openrouter",
            "gemini",
            "openai_compatible",
        ):
            assert provider in PROVIDER_DEFAULTS
            assert "/chat/completions" in PROVIDER_DEFAULTS[provider]["chat_path"]


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_global_semaphore_throttles(self, patch_post):
        from teff.harness import Harness, set_provider_concurrency

        set_provider_concurrency("openai", 2)
        try:
            h = Harness(model="gpt-4", provider="openai", api_key_env="X")
            h._headers = {}
            await asyncio.gather(
                *[h.call([{"role": "user", "content": "x"}]) for _ in range(6)]
            )
            assert patch_post["max_active"] <= 2
        finally:
            set_provider_concurrency("openai", 0)

    def test_set_and_get_concurrency(self):
        from teff.harness import provider_concurrency, set_provider_concurrency

        set_provider_concurrency("openai", 3)
        assert provider_concurrency("openai") == 3
        set_provider_concurrency("openai", 0)
        assert provider_concurrency("openai") is None

    def test_max_parallel_grows_shared_semaphore(self):
        from teff.harness import Harness, provider_concurrency

        Harness(model="gpt-4", provider="openai", api_key_env="X", max_parallel=4)
        assert provider_concurrency("openai") == 4
        Harness(model="gpt-4", provider="openai", api_key_env="X", max_parallel=6)
        assert provider_concurrency("openai") == 6
        from teff.harness import set_provider_concurrency

        set_provider_concurrency("openai", 0)


class TestFromConfigNumbers:
    def test_from_config_honors_zero_max_retries(self):
        from teff.harness import Harness

        h = Harness.from_config(
            {
                "model": "gpt-4",
                "provider": "openai",
                "api_key_env": "X",
                "max_retries": 0,
            }
        )
        assert h.max_retries == 0

    def test_from_config_honors_zero_max_tool_rounds(self):
        from teff.harness import Harness

        h = Harness.from_config(
            {
                "model": "gpt-4",
                "provider": "openai",
                "api_key_env": "X",
                "max_tool_rounds": 0,
            }
        )
        assert h.max_rounds == 0

    def test_from_config_default_max_retries(self):
        from teff.harness import Harness

        h = Harness.from_config(
            {"model": "gpt-4", "provider": "openai", "api_key_env": "X"}
        )
        assert h.max_retries == 2

    def test_stream_token_empty_choices(self):
        from teff.harness import Harness

        h = Harness(model="gpt-4", provider="openai", api_key_env="X")
        assert h._stream_token({"choices": []}) == ""


class TestCostReporting:
    def test_model_pricing_lookup(self):
        from teff.trace import model_pricing, tokens_cost

        assert model_pricing("gpt-4o") == (2.5, 10.0)
        assert model_pricing("gpt-4o-2024-08-06") == (2.5, 10.0)
        assert model_pricing("llama3.1:8b") == (0.0, 0.0)
        # 1M prompt + 1M completion on gpt-4o = 2.5 + 10
        assert tokens_cost("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.5)

    def test_run_summary_cost_and_json(self):
        from teff.trace import RunTracer

        tracer = RunTracer()
        tracer.llm("openai", "gpt-4o", 1000, 2000, 10.0)
        summary = tracer.summary()
        assert summary.cost_usd == pytest.approx(0.0025 + 0.02)
        assert summary.models["gpt-4o"]["prompt_tokens"] == 1000
        d = summary.to_dict()
        assert d["cost_usd"] == summary.cost_usd
        assert "tokens" in d and "total" in d["tokens"]

    def test_trace_json_redacts_secrets(self):
        from teff.trace import RunTracer

        tracer = RunTracer()
        tracer.run_start()
        tracer.node_error(
            "n",
            "llm_chat",
            5.0,
            RuntimeError("boom at https://x?api_key=sk-abcdef12345"),
        )
        text = tracer.to_json()
        assert "sk-abcdef12345" not in text
        assert "api_key=***" in text

    def test_redacted_node_error_keeps_message(self):
        from teff.trace import RunTracer

        tracer = RunTracer()
        tracer.run_start()
        tracer.node_error("n", "llm_chat", 1.0, RuntimeError("plain failure"))
        assert "plain failure" in tracer.to_json()


class TestCustomPricing:
    def teardown_method(self):
        from teff.trace import clear_pricing

        clear_pricing()

    def test_set_model_pricing_overrides_builtin(self):
        from teff.trace import model_pricing, set_model_pricing, tokens_cost

        set_model_pricing("openai", "gpt-4o", 99.0, 199.0)
        assert model_pricing("gpt-4o", "openai") == (99.0, 199.0)
        assert tokens_cost("gpt-4o", 1_000_000, 1_000_000, provider="openai") == (
            99.0 + 199.0
        )
        # built-in table untouched for other providers
        assert model_pricing("gpt-4o") == (2.5, 10.0)

    def test_custom_provider_own_pricing(self):
        from teff.trace import model_pricing, set_model_pricing

        # OpenRouter-style names and rates live outside the built-in table.
        set_model_pricing("openrouter", "openai/gpt-4o", 3.0, 12.0)
        assert model_pricing("openai/gpt-4o", "openrouter") == (3.0, 12.0)
        # unknown provider + unknown model is still free
        assert model_pricing("openai/gpt-4o") == (0.0, 0.0)

    def test_provider_default_price(self):
        from teff.trace import (
            model_pricing,
            set_model_pricing,
            set_provider_pricing,
        )

        set_provider_pricing("kilo", 0.1, 0.4)
        assert model_pricing("any-model", "kilo") == (0.1, 0.4)
        # per-model custom wins over the provider default
        set_model_pricing("kilo", "any-model", 1.0, 2.0)
        assert model_pricing("any-model", "kilo") == (1.0, 2.0)

    def test_prefix_match_within_provider(self):
        from teff.trace import model_pricing, set_model_pricing

        set_model_pricing("openrouter", "anthropic/claude-3.5-sonnet", 3.0, 15.0)
        assert model_pricing("anthropic/claude-3.5-sonnet-20240620", "openrouter") == (
            3.0,
            15.0,
        )

    def test_load_pricing_from_dict(self):
        from teff.trace import load_pricing, model_pricing

        load_pricing(
            {
                "openrouter": {
                    "default": {"input": 0.1, "output": 0.4},
                    "models": {
                        "openai/gpt-4o": {"input": 3.0, "output": 12.0},
                        "deepseek/deepseek-chat": [0.14, 0.28],
                    },
                }
            }
        )
        assert model_pricing("openai/gpt-4o", "openrouter") == (3.0, 12.0)
        assert model_pricing("deepseek/deepseek-chat", "openrouter") == (0.14, 0.28)
        assert model_pricing("something-else", "openrouter") == (0.1, 0.4)

    def test_load_pricing_from_file(self, tmp_path):
        from teff.trace import load_pricing, model_pricing

        path = tmp_path / "pricing.yaml"
        path.write_text(
            "providers:\n"
            "  kilo:\n"
            "    default: {input: 0.05, output: 0.15}\n"
            "    models:\n"
            '      "kilo/mega": {input: 1.0, output: 2.0}\n'
        )
        load_pricing(str(path))
        assert model_pricing("kilo/mega", "kilo") == (1.0, 2.0)
        assert model_pricing("anything", "kilo") == (0.05, 0.15)

    def test_summary_uses_provider_pricing(self):
        from teff.trace import RunTracer, set_model_pricing

        set_model_pricing("openrouter", "openai/gpt-4o", 3.0, 12.0)
        tracer = RunTracer()
        tracer.llm("openrouter", "openai/gpt-4o", 1_000_000, 1_000_000, 10.0)
        summary = tracer.summary()
        # 3.0 + 12.0 (custom), not the built-in gpt-4o price of 2.5 + 10
        assert summary.cost_usd == pytest.approx(15.0)
