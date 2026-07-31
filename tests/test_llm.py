import pytest


class TestParseTextToolCall:
    def test_parses_parameters_form(self):
        from draf.node.llm import _parse_text_tool_call

        content = 'Calling tool: {"name": "rag", "parameters": {"k": 5, "query": "x"}}'
        assert _parse_text_tool_call(content) == ("rag", {"k": 5, "query": "x"})

    def test_parses_arguments_form(self):
        from draf.node.llm import _parse_text_tool_call

        content = '{"name": "calc", "arguments": {"expression": "2+2"}}'
        assert _parse_text_tool_call(content) == ("calc", {"expression": "2+2"})

    def test_no_tool_call_returns_none(self):
        from draf.node.llm import _parse_text_tool_call

        assert _parse_text_tool_call("just a normal answer") is None

    def test_nested_braces_in_arguments(self):
        from draf.node.llm import _parse_text_tool_call

        content = '{"name": "rag", "parameters": {"k": 1, "query": "a {b} c"}}'
        assert _parse_text_tool_call(content) == (
            "rag",
            {"k": 1, "query": "a {b} c"},
        )


class TestLLMNode:
    @pytest.mark.asyncio
    async def test_sends_request_and_stores_response(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "hi there"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = LLM({"model": "gpt-4", "system": "Say hi", "output_key": "greeting"})
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {"greeting": ""})
        assert result["greeting"] == "hi there"

    @pytest.mark.asyncio
    async def test_prompt_template_renders_state_keys(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        captured = {}

        async def mock_post(*a, **kw):
            captured["body"] = kw.get("json")

            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "plan"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = LLM(
            {
                "model": "gpt-4",
                "system": "Вы инженер",
                "prompt": "составь план для ремонта {type} на сумму {summ}",
                "output_key": "plan",
            }
        )
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {"type": "кухни", "summ": 15000})
        assert result["plan"] == "plan"

        messages = captured["body"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Вы инженер"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "составь план для ремонта кухни на сумму 15000"

    @pytest.mark.asyncio
    async def test_prompt_template_missing_key_raises(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        node = LLM({"model": "gpt-4", "prompt": "ремонт {type} на {summ}"})
        ctx = ExecContext(state={}, tools={})
        with pytest.raises(KeyError, match="unknown state key"):
            await node.execute(ctx, {"type": "кухни"})

    @pytest.mark.asyncio
    async def test_structured_output_json_mode(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": '{"ok": true}'}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = LLM({"model": "gpt-4", "response_format": {"type": "json_object"}})
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {})
        assert result["output"] == '{"ok": true}'

    @pytest.mark.asyncio
    async def test_tool_calling_loop(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext
        from draf.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class UppercaseTool(Tool):
            name = "uppercase"
            description = "Convert text to uppercase"

            def run(self, text: str = "") -> str:  # type: ignore[override]
                return text.upper()

        tool_call_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "uppercase",
                                    "arguments": '{"text": "hello"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
        content_response = {
            "choices": [
                {
                    "message": {"content": "the result is HELLO"},
                }
            ],
        }

        responses = [tool_call_response, content_response]

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return responses.pop(0)

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = LLM({"model": "gpt-4", "use_tools": True})
        ctx = ExecContext(state={}, tools={"uppercase": UppercaseTool()})
        result = await node.execute(ctx, {})
        assert result["output"] == "the result is HELLO"

    @pytest.mark.asyncio
    async def test_tool_calling_executes_and_feeds_back(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext
        from draf.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        call_log = []

        class LogTool(Tool):
            name = "logger"
            description = "Log a message"

            def run(self, msg: str = "") -> str:  # type: ignore[override]
                call_log.append(msg)
                return f"logged: {msg}"

        tool_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "logger",
                                    "arguments": '{"msg": "test"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
        content_resp = {
            "choices": [{"message": {"content": "done"}}],
        }

        responses = [tool_resp, content_resp]

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return responses.pop(0)

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = LLM({"model": "gpt-4", "use_tools": True})
        ctx = ExecContext(state={}, tools={"logger": LogTool()})
        result = await node.execute(ctx, {})
        assert result["output"] == "done"
        assert call_log == ["test"]

    @pytest.mark.asyncio
    async def test_manual_tool_defs_without_use_tools(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = LLM(
            {
                "model": "gpt-4",
                "tools": [{"type": "function", "function": {"name": "x"}}],
            }
        )
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {})
        assert result["output"] == "ok"

    @pytest.mark.asyncio
    async def test_tool_schema_generation(self):
        from draf.node import LLM
        from draf.tool import Tool

        class MyTool(Tool):
            name = "mytool"
            description = "Does stuff"

            def run(self, x: int = 0, y: str = "") -> str:  # type: ignore[override]
                return f"{x} {y}"

        schema = LLM._tool_to_schema(MyTool())
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mytool"
        assert "x" in schema["function"]["parameters"]["properties"]
        assert "y" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == []

    @pytest.mark.asyncio
    async def test_tool_schema_required_params(self):
        from draf.node import LLM
        from draf.tool import Tool

        class ReqTool(Tool):
            name = "req"
            description = "Required params"

            def run(self, name: str, age: int) -> str:  # type: ignore[override]
                return f"{name} {age}"

        schema = LLM._tool_to_schema(ReqTool())
        assert set(schema["function"]["parameters"]["required"]) == {"name", "age"}

    @pytest.mark.asyncio
    async def test_streaming_accumulates_content(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            "data: [DONE]\n",
        ]

        class MockStreamResponse:
            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

        class MockStreamCM:
            async def __aenter__(self):
                return MockStreamResponse()

            async def __aexit__(self, *a):
                pass

        def mock_stream(*a, **kw):
            return MockStreamCM()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

        node = LLM({"model": "gpt-4", "stream": True})
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {})
        assert result["output"] == "Hello world"

    @pytest.mark.asyncio
    async def test_streaming_calls_on_token(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}\n',
            'data: {"choices":[{"delta":{"content":"b"}}]}\n',
            "data: [DONE]\n",
        ]

        class MockStreamResponse:
            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

        class MockStreamCM:
            async def __aenter__(self):
                return MockStreamResponse()

            async def __aexit__(self, *a):
                pass

        def mock_stream(*a, **kw):
            return MockStreamCM()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

        tokens = []

        def on_token(t: str) -> None:
            tokens.append(t)

        node = LLM({"model": "gpt-4", "stream": True, "on_token": on_token})
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {})
        assert result["output"] == "ab"
        assert tokens == ["a", "b"]

    @pytest.mark.asyncio
    async def test_streaming_disabled_when_tools_are_used(self, monkeypatch):
        from draf.node import LLM
        from draf.node import ExecContext
        from draf.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "done"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        class SimpleTool(Tool):
            name = "simple"
            description = "simple"

            def run(self) -> str:  # type: ignore[override]
                return "ok"

        node = LLM({"model": "gpt-4", "stream": True, "use_tools": True})
        ctx = ExecContext(state={}, tools={"simple": SimpleTool()})
        result = await node.execute(ctx, {})
        assert result["output"] == "done"
