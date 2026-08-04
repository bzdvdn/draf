import pytest

from draf.provider import ProviderRegistry


class TestFakeLLM:
    @pytest.mark.asyncio
    async def test_returns_canned_content(self):
        from draf.node import ExecContext
        from draf.testing import FakeLLM

        node = FakeLLM({"output_key": "reply", "content": "hello"})
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {"name": "world"})
        assert result == {"reply": "hello"}

    @pytest.mark.asyncio
    async def test_renders_template_from_state(self):
        from draf.node import ExecContext
        from draf.testing import FakeLLM

        node = FakeLLM({"prompt": "greet {name}", "content": "hello {name}"})
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {"name": "Ana"})
        assert result == {"output": "hello Ana"}

    @pytest.mark.asyncio
    async def test_missing_prompt_key_raises(self):
        from draf.node import ExecContext
        from draf.testing import FakeLLM

        node = FakeLLM({"prompt": "greet {name}"})
        ctx = ExecContext(state={}, tools={})
        with pytest.raises(KeyError):
            await node.execute(ctx, {})

    @pytest.mark.asyncio
    async def test_works_inside_graph(self):
        from draf.graph import Graph
        from draf.testing import FakeLLM

        g = Graph(
            nodes={"a": FakeLLM({"content": "hi"})},
            edges=[],
            entry_point="a",
        )
        result = await g.run(state={})
        assert result["output"] == "hi"

    @pytest.mark.asyncio
    async def test_default_output_key_and_content(self):
        from draf.node import ExecContext
        from draf.testing import FakeLLM

        node = FakeLLM({})
        ctx = ExecContext(state={}, tools={})
        assert await node.execute(ctx, {}) == {"output": "mock"}


class TestMockLLM:
    @pytest.mark.asyncio
    async def test_mock_llm_fixture_patches_harness(self, mock_llm):
        from draf.node import LLM, ExecContext

        mock_llm.content = "42"

        node = LLM(
            {
                "model": "gpt-4",
                "prompt": "calc",
                "output_key": "answer",
                "provider": "openai",
            }
        )
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {})
        assert result["answer"] == "42"
        assert len(mock_llm.calls) == 1
        assert mock_llm.calls[0]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_records_prompt_messages(self, mock_llm):
        from draf.node import LLM, ExecContext

        node = LLM(
            {"model": "gpt-4", "system": "sys", "prompt": "ask", "provider": "openai"}
        )
        ctx = ExecContext(state={}, tools={})
        await node.execute(ctx, {})
        messages = mock_llm.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "sys"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "ask"

    @pytest.mark.asyncio
    async def test_structured_output_parses_canned_json(self, mock_llm):
        from draf.node import LLM, ExecContext
        from draf.testing import canned_json

        mock_llm.content = canned_json({"answer": 42, "ok": True})

        node = LLM(
            {
                "model": "gpt-4",
                "prompt": "return json",
                "output_type": {"type": "object", "properties": {}},
                "provider": "openai",
            }
        )
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {})
        assert result["output"] == {"answer": 42, "ok": True}

    @pytest.mark.asyncio
    async def test_react_agent_runs_offline(self, mock_llm):
        from draf.graph import Graph
        from draf.node import ReActAgent

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {
                        "model": "gpt-4",
                        "prompt": "answer",
                        "system": "be brief",
                        "provider": "openai",
                    }
                )
            },
            edges=[],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
        )
        result = await g.run(state={})
        assert result["output"] == "mock"
        assert mock_llm.calls

    @pytest.mark.asyncio
    async def test_mock_llm_no_network(self, mock_llm):
        from draf.graph import Graph
        from draf.node import LLM

        g = Graph(
            nodes={"a": LLM({"model": "gpt-4", "prompt": "hi", "provider": "openai"})},
            edges=[],
            entry_point="a",
            providers=ProviderRegistry.from_presets("openai"),
        )
        result = await g.run(state={})
        assert result["output"] == "mock"
