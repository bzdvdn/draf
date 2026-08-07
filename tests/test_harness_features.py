"""Tests for harness/agent production features: retries, failover, timeouts,
context management, token budget, tool approval, nested schemas, streaming."""

import json

import pytest

from teff.provider import ProviderRegistry


def _mock_response(data: dict):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    return MockResponse()


class TestNestedSchema:
    def test_flat_types(self):
        from teff.harness import tool_to_schema
        from teff.tool import Tool

        class Flat(Tool):
            name = "flat"
            description = "flat tool"

            def run(self, text: str = "", count: int = 0) -> str:  # type: ignore[override]
                return text

        schema = tool_to_schema(Flat())
        props = schema["function"]["parameters"]["properties"]
        assert props["text"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        # defaults are preserved
        assert props["text"]["default"] == ""
        assert props["count"]["default"] == 0

    def test_nested_list_of_dict(self):
        from teff.harness import tool_to_schema
        from teff.tool import Tool

        class Nested(Tool):
            name = "nested"
            description = "nested tool"

            def run(self, rows: list[dict] | None = None) -> str:  # type: ignore[override]
                return ""

        schema = tool_to_schema(Nested())
        prop = schema["function"]["parameters"]["properties"]["rows"]
        # Optional union expands to oneOf: [null, array]; default=None is dropped
        assert "oneOf" in prop
        array_branch = [b for b in prop["oneOf"] if b.get("type") == "array"]
        assert array_branch

    def test_nested_typed_list(self):
        from typing import TypedDict

        from teff.harness import tool_to_schema
        from teff.tool import Tool

        class Item(TypedDict):
            name: str
            qty: int

        class Cart(Tool):
            name = "cart"
            description = "cart tool"

            def run(self, items: list[Item]) -> str:  # type: ignore[override]
                return ""

        schema = tool_to_schema(Cart())
        prop = schema["function"]["parameters"]["properties"]["items"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "object"
        assert set(prop["items"]["properties"].keys()) == {"name", "qty"}
        assert "items" in schema["function"]["parameters"]["required"]

    def test_dict_str_to_str(self):
        from teff.harness import tool_to_schema
        from teff.tool import Tool

        class Map(Tool):
            name = "map_tool"
            description = "map tool"

            def run(self, env: dict[str, str]) -> str:  # type: ignore[override]
                return ""

        schema = tool_to_schema(Map())
        props = schema["function"]["parameters"]["properties"]
        assert props["env"]["type"] == "object"
        assert "env" in schema["function"]["parameters"]["required"]

    def test_typeddict_expands(self):
        from typing import TypedDict

        from teff.harness import tool_to_schema
        from teff.tool import Tool

        class Config(TypedDict):
            host: str
            port: int

        class CfgTool(Tool):
            name = "cfg_tool"
            description = "cfg tool"

            def run(self, config: Config) -> str:  # type: ignore[override]
                return ""

        schema = tool_to_schema(CfgTool())
        prop = schema["function"]["parameters"]["properties"]["config"]
        assert prop["type"] == "object"
        assert set(prop["properties"].keys()) == {"host", "port"}


class TestHttpRetry:
    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, monkeypatch):
        import httpx

        from teff.harness import Harness

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        calls = {"n": 0}

        class FlakyResponse:
            def raise_for_status(self):
                if calls["n"] < 2:
                    calls["n"] += 1
                    req = httpx.Request("POST", "http://x")
                    raise httpx.HTTPStatusError(
                        "500", request=req, response=httpx.Response(500, request=req)
                    )
                return None

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        async def mock_post(*a, **kw):
            return FlakyResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        h = Harness(model="gpt-4", provider="openai", max_retries=3, retry_on=(500,))
        reply = await h.call([{"role": "user", "content": "hi"}])
        assert reply.content == "ok"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, monkeypatch):
        import httpx

        from teff.harness import Harness

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        calls = {"n": 0}

        class AlwaysFail:
            def raise_for_status(self):
                calls["n"] += 1
                raise httpx.ConnectError("down")

            def json(self):
                return {}

        async def mock_post(*a, **kw):
            return AlwaysFail()

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        h = Harness(model="gpt-4", provider="openai", max_retries=2)
        with pytest.raises(httpx.ConnectError):
            await h.call([{"role": "user", "content": "hi"}])
        assert calls["n"] == 3  # initial + 2 retries


class TestFailover:
    @pytest.mark.asyncio
    async def test_falls_back_to_secondary_model(self, monkeypatch):
        import httpx

        from teff.harness import Harness

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        calls = {"n": 0}
        bodies = []

        class BoomResponse:
            def raise_for_status(self):
                calls["n"] += 1
                raise httpx.ConnectError("primary down")

            def json(self):
                return {}

        async def mock_post(url, *a, **kw):
            bodies.append({"url": url, "json": kw.get("json")})
            return BoomResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        h = Harness(
            model="gpt-4",
            provider="openai",
            max_retries=0,
            fallbacks=["gpt-4-turbo"],
            base_url="http://primary/v1",
            chat_path="/chat/completions",
        )
        # both primary and fallback fail; assert fallback transport was attempted
        with pytest.raises(httpx.ConnectError):
            await h.call([{"role": "user", "content": "hi"}])
        assert calls["n"] == 2  # primary + fallback
        models = [b["json"]["model"] for b in bodies]
        assert models == ["gpt-4", "gpt-4-turbo"]


class TestToolTimeoutAndRetries:
    @pytest.mark.asyncio
    async def test_tool_timeout_returns_error_message(self):
        import asyncio

        from teff.harness import execute_tool_calls
        from teff.tool import Tool

        class Slow(Tool):
            name = "slow"
            description = "slow tool"

            async def arun(self, **kwargs):
                await asyncio.sleep(5)
                return "done"

        results = await execute_tool_calls(
            [{"id": "c1", "function": {"name": "slow", "arguments": "{}"}}],
            {"slow": Slow()},
            timeout=0.05,
        )
        assert "timed out" in results[0]

    @pytest.mark.asyncio
    async def test_tool_retries_flaky_tool(self):
        from teff.harness import execute_tool_calls
        from teff.tool import Tool

        class Flaky(Tool):
            name = "flaky"
            description = "flaky tool"
            attempts = 0

            def run(self, **kwargs) -> str:
                Flaky.attempts += 1
                if Flaky.attempts < 3:
                    raise RuntimeError("boom")
                return "recovered"

        results = await execute_tool_calls(
            [{"id": "c1", "function": {"name": "flaky", "arguments": "{}"}}],
            {"flaky": Flaky()},
            tool_retries=2,
        )
        assert results == ["recovered"]
        assert Flaky.attempts == 3


class TestContextManagement:
    def test_trim_messages_keeps_system(self):
        from teff.harness import trim_messages

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        trimmed = trim_messages(msgs, max_messages=3)
        assert trimmed[0]["role"] == "system"
        assert [m["role"] for m in trimmed] == [
            "system",
            "assistant",
            "user",
            "assistant",
        ]
        assert trimmed[-1]["content"] == "d"

    def test_trim_by_tokens(self):
        from teff.harness import trim_messages

        long = "x" * 400  # ~100 tokens
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": long},
            {"role": "assistant", "content": "answer"},
        ]
        trimmed = trim_messages(msgs, max_tokens=60)
        assert trimmed[0]["role"] == "system"
        assert len(trimmed) == 2
        assert trimmed[-1]["role"] == "assistant"


class TestTokenBudget:
    @pytest.mark.asyncio
    async def test_run_stops_on_budget(self, monkeypatch):
        import httpx

        from teff.harness import Harness
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class AlwaysTool(Tool):
            name = "always"
            description = "always tool"

            def run(self) -> str:  # type: ignore[override]
                return "done"

        tool_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "always", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 50},
        }

        calls = {"n": 0}

        async def mock_post(*a, **kw):
            calls["n"] += 1
            return _mock_response(tool_resp)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        h = Harness(
            model="gpt-4", provider="openai", max_rounds=10, max_total_tokens=150
        )
        step = await h.run(
            [{"role": "user", "content": "go"}],
            {"always": AlwaysTool()},
        )
        assert step.wants_tool
        assert h.total_tokens >= 150
        assert (
            calls["n"] == 2
        )  # one more after budget reached? no: stop check before call


class TestToolApproval:
    @pytest.mark.asyncio
    async def test_deny_shortcircuits_call(self, monkeypatch):
        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        executed = {"n": 0}

        class Sensitive(Tool):
            name = "sensitive"
            description = "sensitive tool"

            def run(self, cmd: str = "") -> str:  # type: ignore[override]
                executed["n"] += 1
                return f"ran {cmd}"

        tool_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "sensitive",
                                    "arguments": '{"cmd": "rm -rf /"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }
        final_resp = {"choices": [{"message": {"content": "final"}}]}
        responses = [tool_resp, final_resp]

        async def mock_post(*a, **kw):
            return _mock_response(responses.pop(0))

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "agent": ReActAgent({"model": "gpt-4", "input_key": "input"}),
                "tool": ToolExec({"tool_approval": lambda name, args: False}),
            },
            edges=[
                Edge("agent", "tool", "_tool_call_name!="),
                Edge("tool", "agent"),
            ],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        r = await g.run(
            state={"input": "delete files"},
            tools=[Sensitive()],
            max_iterations=5,
        )
        assert r["output"] == "final"
        assert executed["n"] == 0
        tool_msgs = [m for m in r["messages"] if m.get("role") == "tool"]
        assert "not approved" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_approve_allows_execution(self, monkeypatch):
        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class Safe(Tool):
            name = "safe"
            description = "safe tool"

            def run(self, text: str = "") -> str:  # type: ignore[override]
                return text.upper()

        tool_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "safe",
                                    "arguments": '{"text": "hello"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }
        final_resp = {"choices": [{"message": {"content": "HELLO"}}]}
        responses = [tool_resp, final_resp]

        async def mock_post(*a, **kw):
            return _mock_response(responses.pop(0))

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {"model": "gpt-4", "input_key": "input", "use_tools": ["safe"]}
                ),
                "tool": ToolExec(
                    {"tool_approval": lambda name, args: True, "use_tools": ["safe"]}
                ),
            },
            edges=[
                Edge("agent", "tool", "_tool_call_name!="),
                Edge("tool", "agent"),
            ],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        r = await g.run(state={"input": "hi"}, tools=[Safe()], max_iterations=5)
        assert r["output"] == "HELLO"
        tool_msgs = [m for m in r["messages"] if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "HELLO"

    @pytest.mark.asyncio
    async def test_pause_raises_interrupt(self, monkeypatch):
        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec
        from teff.node.interrupt import GraphInterrupt
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class Sensitive(Tool):
            name = "sensitive"
            description = "sensitive tool"

            def run(self, cmd: str = "") -> str:  # type: ignore[override]
                return f"ran {cmd}"

        tool_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "sensitive",
                                    "arguments": '{"cmd": "danger"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }

        async def mock_post(*a, **kw):
            return _mock_response(tool_resp)

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        async def approver(name, args):
            return "pause"

        g = Graph(
            nodes={
                "agent": ReActAgent({"model": "gpt-4", "input_key": "input"}),
                "tool": ToolExec({"tool_approval": approver}),
            },
            edges=[
                Edge("agent", "tool", "_tool_call_name!="),
                Edge("tool", "agent"),
            ],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        with pytest.raises(GraphInterrupt) as excinfo:
            await g.run(
                state={"input": "do it"},
                tools=[Sensitive()],
                max_iterations=5,
            )
        assert excinfo.value.key == "tool_approval"

    @pytest.mark.asyncio
    async def test_pause_then_resume_approves(self, monkeypatch, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer
        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec
        from teff.node.interrupt import GraphInterrupt
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class Sensitive(Tool):
            name = "sensitive"
            description = "sensitive tool"

            def run(self, cmd: str = "") -> str:  # type: ignore[override]
                return f"ran {cmd}"

        tool_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "sensitive",
                                    "arguments": '{"cmd": "danger"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }
        final_resp = {"choices": [{"message": {"content": "approved answer"}}]}
        # First run: agent emits the tool call, then pauses for approval.
        # On resume the graph continues past the interrupt (tool -> agent), so
        # the agent re-plans and re-emits the tool call before ToolExec
        # consumes the resume decision and actually executes the tool.
        responses = [tool_resp, tool_resp, final_resp]

        async def mock_post(*a, **kw):
            return _mock_response(responses.pop(0))

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        async def approver(name, args):
            return "pause"

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {"model": "gpt-4", "input_key": "input", "use_tools": ["sensitive"]}
                ),
                "tool": ToolExec(
                    {"tool_approval": approver, "use_tools": ["sensitive"]}
                ),
            },
            edges=[
                Edge("agent", "tool", "_tool_call_name!="),
                Edge("tool", "agent"),
            ],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        cp = JSONFileCheckpointer(str(tmp_path / "cp.json"))
        state = {"input": "do it"}
        with pytest.raises(GraphInterrupt):
            await g.run(
                state=state,
                tools=[Sensitive()],
                max_iterations=5,
                checkpointer=cp,
                checkpoint_id="run-1",
            )
        result = await g.run(
            state=state,
            tools=[Sensitive()],
            max_iterations=5,
            checkpointer=cp,
            checkpoint_id="run-1",
            resume={"tool_approval": "approve"},
        )
        assert result["output"] == "approved answer"
        tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
        assert tool_msgs and tool_msgs[0]["content"] == "ran danger"

    @pytest.mark.asyncio
    async def test_harness_on_tool_call_hook(self, monkeypatch):
        import httpx

        from teff.harness import Harness
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        seen = []

        class Echo(Tool):
            name = "echo"
            description = "echo tool"

            def run(self, text: str = "") -> str:  # type: ignore[override]
                return text

        tool_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"text": "hi"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }
        final_resp = {"choices": [{"message": {"content": "done"}}]}
        responses = [tool_resp, final_resp]

        async def mock_post(*a, **kw):
            return _mock_response(responses.pop(0))

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        h = Harness(model="gpt-4", provider="openai")

        async def hook(name, args):
            seen.append((name, args))

        h.on_tool_call = hook
        step = await h.run(
            [{"role": "user", "content": "go"}],
            {"echo": Echo()},
        )
        assert seen == [("echo", {"text": "hi"})]
        assert step.content == "done"


class TestReActStreaming:
    @pytest.mark.asyncio
    async def test_agent_streams_tokens(self, monkeypatch):
        from teff.graph import Graph
        from teff.node.agent import ReActAgent

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class StreamingResponse:
            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def aiter_lines(self):
                for tok in ("hel", "lo ", "world"):
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': tok}}]})}"
                yield "data: [DONE]"

        def mock_stream(*a, **kw):
            return StreamingResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

        tokens = []

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {
                        "model": "gpt-4",
                        "input_key": "input",
                        "output_key": "output",
                        "stream": True,
                        "on_token": tokens.append,
                    }
                )
            },
            edges=[],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        r = await g.run(state={"input": "hi"})
        assert r["output"] == "hello world"
        assert tokens == ["hel", "lo ", "world"]
