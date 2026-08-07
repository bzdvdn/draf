import pytest

from teff.provider import ProviderRegistry


class TestReActAgent:
    @pytest.mark.asyncio
    async def test_agent_responds_directly(self, monkeypatch):
        from teff.graph import Graph
        from teff.node.agent import ReActAgent

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "hello world"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {"model": "gpt-4", "input_key": "input", "output_key": "output"}
                )
            },
            edges=[],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        r = await g.run(state={"input": "hi"})
        assert r["output"] == "hello world"
        assert r.get("_tool_call_name") == ""

    @pytest.mark.asyncio
    async def test_agent_tool_cycle(self, monkeypatch):
        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class UpperTool(Tool):
            name = "uppercase"
            description = "Make it uppercase"

            def run(self, text: str = "") -> str:  # type: ignore[override]
                return text.upper()

        tool_response = {
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
        final_response = {
            "choices": [{"message": {"content": "Result: HELLO"}}],
        }

        responses = [tool_response, final_response]

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return responses.pop(0)

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {"model": "gpt-4", "input_key": "input", "output_key": "output"}
                ),
                "tool": ToolExec({}),
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
            state={"input": "make it uppercase"}, tools=[UpperTool()], max_iterations=5
        )
        assert r["output"] == "Result: HELLO"
        assert r["_tool_call_name"] == ""
        assert len(r["messages"]) == 4  # user + assistant(tool_call) + tool + assistant

    @pytest.mark.asyncio
    async def test_agent_tool_cycle_append_reducer_no_duplicates(self, monkeypatch):
        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class UpperTool(Tool):
            name = "uppercase"
            description = "Make it uppercase"

            def run(self, text: str = "") -> str:  # type: ignore[override]
                return text.upper()

        tool_response = {
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
        final_response = {
            "choices": [{"message": {"content": "Result: HELLO"}}],
        }

        responses = [tool_response, final_response]

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return responses.pop(0)

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {"model": "gpt-4", "input_key": "input", "output_key": "output"}
                ),
                "tool": ToolExec({}),
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
            state={"input": "make it uppercase"},
            tools=[UpperTool()],
            reducers={"messages": "append"},
            max_iterations=5,
        )
        assert r["output"] == "Result: HELLO"
        assert r["_tool_call_name"] == ""
        assert len(r["messages"]) == 4  # no duplication under append reducer

    @pytest.mark.asyncio
    async def test_agent_loop_exceeds_max_iterations(self, monkeypatch):
        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        tool_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "always_tool", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
        }

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return tool_response

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        from teff.tool import Tool

        class AlwaysTool(Tool):
            name = "always_tool"
            description = "Always called"

            def run(self) -> str:  # type: ignore[override]
                return "done"

        g = Graph(
            nodes={
                "agent": ReActAgent({"model": "gpt-4", "input_key": "input"}),
                "tool": ToolExec({}),
            },
            edges=[
                Edge("agent", "tool", "_tool_call_name!="),
                Edge("tool", "agent"),
            ],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        with pytest.raises(RuntimeError, match="max_iterations"):
            await g.run(
                state={"input": "do it"}, tools=[AlwaysTool()], max_iterations=3
            )


class TestParallelToolExecution:
    @pytest.mark.asyncio
    async def test_multiple_tools_fan_out_in_parallel(self, monkeypatch):
        import asyncio

        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        lock = asyncio.Lock()
        tracker = {"active": 0, "max_active": 0}

        class SlowTool(Tool):
            name = "slowtool"
            description = "Slow async tool"

            async def arun(self, **kwargs):  # type: ignore[override]
                async with lock:
                    tracker["active"] += 1
                    tracker["max_active"] = max(
                        tracker["max_active"], tracker["active"]
                    )
                await asyncio.sleep(0.05)
                async with lock:
                    tracker["active"] -= 1
                return "slow-done"

        tool_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "slowtool", "arguments": "{}"},
                            },
                            {
                                "id": "call_2",
                                "function": {"name": "slowtool", "arguments": "{}"},
                            },
                        ],
                    },
                }
            ],
        }
        final_response = {
            "choices": [{"message": {"content": "finished"}}],
        }

        responses = [tool_response, final_response]

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return responses.pop(0)

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {
                        "model": "gpt-4",
                        "input_key": "input",
                        "output_key": "output",
                        "use_tools": ["slowtool"],
                    }
                ),
                "tool": ToolExec({"use_tools": ["slowtool"]}),
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
            state={"input": "do two things"}, tools=[SlowTool()], max_iterations=10
        )
        assert r["output"] == "finished"
        assert tracker["max_active"] == 2
        assert len(r["messages"]) == 5  # user + assistant(tc) + 2 tool + assistant
        tool_msgs = [m for m in r["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert all(m["content"] == "slow-done" for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_single_round_parallel_saves_a_round(self, monkeypatch):
        import asyncio

        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        lock = asyncio.Lock()
        tracker = {"active": 0, "max_active": 0}

        class SlowTool(Tool):
            name = "slowtool"
            description = "Slow async tool"

            async def arun(self, **kwargs):  # type: ignore[override]
                async with lock:
                    tracker["active"] += 1
                    tracker["max_active"] = max(
                        tracker["max_active"], tracker["active"]
                    )
                await asyncio.sleep(0.1)
                async with lock:
                    tracker["active"] -= 1
                return "slow-done"

        tool_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "slowtool", "arguments": "{}"},
                            },
                            {
                                "id": "call_2",
                                "function": {"name": "slowtool", "arguments": "{}"},
                            },
                        ],
                    },
                }
            ],
        }
        final_response = {
            "choices": [{"message": {"content": "finished"}}],
        }
        responses = [tool_response, final_response]

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return responses.pop(0)

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {"model": "gpt-4", "input_key": "input", "output_key": "output"}
                ),
                "tool": ToolExec({}),
            },
            edges=[
                Edge("agent", "tool", "_tool_call_name!="),
                Edge("tool", "agent"),
            ],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        started = asyncio.get_event_loop().time()
        await g.run(
            state={"input": "do two things"}, tools=[SlowTool()], max_iterations=10
        )
        elapsed = asyncio.get_event_loop().time() - started
        assert elapsed < 0.19  # two 0.1s sleeps would take >= 0.2s if serial


class TestToolErrorMode:
    @pytest.mark.asyncio
    async def test_tool_error_becomes_message_by_default(self, monkeypatch):
        from teff.graph import Edge, Graph
        from teff.node.agent import ReActAgent, ToolExec
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class BoomTool(Tool):
            name = "boom"
            description = "Always fails"

            def run(self, **kwargs) -> str:  # type: ignore[override]
                raise RuntimeError("kaboom")

        tool_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "boom", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
        }
        final_response = {
            "choices": [{"message": {"content": "recovered"}}],
        }
        responses = [tool_response, final_response]

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return responses.pop(0)

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {
                        "model": "gpt-4",
                        "input_key": "input",
                        "output_key": "output",
                        "use_tools": ["boom"],
                    }
                ),
                "tool": ToolExec({"use_tools": ["boom"]}),
            },
            edges=[
                Edge("agent", "tool", "_tool_call_name!="),
                Edge("tool", "agent"),
            ],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        r = await g.run(state={"input": "try"}, tools=[BoomTool()], max_iterations=5)
        assert r["output"] == "recovered"
        tool_msgs = [m for m in r["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "Error calling 'boom'" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_error_raise_routes_to_error_edge(self, monkeypatch):
        from teff.graph import Edge, Graph
        from teff.node import Transform
        from teff.node.agent import ReActAgent, ToolExec
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class BoomTool(Tool):
            name = "boom"
            description = "Always fails"

            def run(self, **kwargs) -> str:  # type: ignore[override]
                raise RuntimeError("kaboom")

        tool_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "boom", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
        }

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return tool_response

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "agent": ReActAgent(
                    {
                        "model": "gpt-4",
                        "input_key": "input",
                        "output_key": "output",
                        "use_tools": ["boom"],
                    }
                ),
                "tool": ToolExec({"tool_error_mode": "raise", "use_tools": ["boom"]}),
                "fallback": Transform(
                    {
                        "action": "value",
                        "value": "fallback-handled",
                        "output_key": "output",
                    }
                ),
            },
            edges=[
                Edge("agent", "tool", "_tool_call_name!="),
                Edge("tool", "agent"),
                Edge("tool", "fallback", "__error__"),
            ],
            entry_point="agent",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        r = await g.run(state={"input": "try"}, tools=[BoomTool()], max_iterations=5)
        assert r["output"] == "fallback-handled"


class TestFlowReact:
    @pytest.mark.asyncio
    async def test_flow_react_direct_response(self, monkeypatch):
        from teff.flow import Flow

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "direct answer"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        flow = Flow(
            "test",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        flow.react(
            model="gpt-4",
            system="You are helpful.",
            input_key="query",
            output_key="answer",
        )
        g = flow.compile()
        r = await g.run(state={"query": "hello"}, max_iterations=5)
        assert r["answer"] == "direct answer"

    @pytest.mark.asyncio
    async def test_flow_react_tool_cycle(self, monkeypatch):
        from teff.flow import Flow
        from teff.tool import Tool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class ReverseTool(Tool):
            name = "reverse"
            description = "Reverse a string"

            def run(self, text: str = "") -> str:  # type: ignore[override]
                return text[::-1]

        tool_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "reverse",
                                    "arguments": '{"text": "hello"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
        final_resp = {
            "choices": [{"message": {"content": "reversed: olleh"}}],
        }

        responses = [tool_resp, final_resp]

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return responses.pop(0)

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        flow = Flow(
            "test",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        flow.react(model="gpt-4", input_key="query", output_key="answer")
        g = flow.compile()
        r = await g.run(
            state={"query": "reverse it"}, tools=[ReverseTool()], max_iterations=10
        )
        assert r["answer"] == "reversed: olleh"
        assert len(r["messages"]) == 4

    @pytest.mark.asyncio
    async def test_flow_react_chains_after(self, monkeypatch):
        from teff.flow import Flow
        from teff.node import Transform

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "hello world"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        flow = Flow(
            "test",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        flow.react(model="gpt-4", input_key="query", output_key="answer")
        flow.step(
            Transform(
                {"action": "uppercase", "input_key": "answer", "output_key": "result"}
            )
        )
        g = flow.compile()
        r = await g.run(state={"query": "hi"}, max_iterations=5)
        assert r["result"] == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_flow_harness_accepts_loop_kwargs(self, monkeypatch):
        from teff.flow import Flow
        from teff.node import Transform

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "hello world"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        flow = Flow(
            "test",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        flow.harness(
            model="gpt-4",
            system="You are helpful.",
            input_key="query",
            output_key="answer",
            max_tool_rounds=3,
            tool_error_mode="raise",
            parse_text_tool_calls=False,
            temperature=0.2,
            max_tokens=64,
        )
        flow.step(
            Transform(
                {"action": "uppercase", "input_key": "answer", "output_key": "result"}
            )
        )
        g = flow.compile()
        r = await g.run(state={"query": "hi"}, max_iterations=10)
        assert r["answer"] == "hello world"
        assert r["result"] == "HELLO WORLD"

        from teff.node.agent import ReActAgent, ToolExec

        agent = next(n for n in g.nodes.values() if isinstance(n, ReActAgent))
        assert agent.config["max_tool_rounds"] == 3
        assert agent.config["parse_text_tool_calls"] is False
        assert agent.config["temperature"] == 0.2
        assert agent.config["max_tokens"] == 64

        tool_exec = next(n for n in g.nodes.values() if isinstance(n, ToolExec))
        assert tool_exec.config["tool_error_mode"] == "raise"
