import pytest


class TestReActAgent:
    @pytest.mark.asyncio
    async def test_agent_responds_directly(self, monkeypatch):
        from draf.node.agent import ReActAgent
        from draf.graph import Graph

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
        )
        r = await g.run(state={"input": "hi"})
        assert r["output"] == "hello world"
        assert r.get("_tool_call_name") == ""

    @pytest.mark.asyncio
    async def test_agent_tool_cycle(self, monkeypatch):
        from draf.node.agent import ReActAgent, ToolExec
        from draf.graph import Graph, Edge
        from draf.tool import Tool

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
        )
        r = await g.run(
            state={"input": "make it uppercase"}, tools=[UpperTool()], max_iterations=5
        )
        assert r["output"] == "Result: HELLO"
        assert r["_tool_call_name"] == ""
        assert len(r["messages"]) == 4  # user + assistant(tool_call) + tool + assistant

    @pytest.mark.asyncio
    async def test_agent_loop_exceeds_max_iterations(self, monkeypatch):
        from draf.node.agent import ReActAgent, ToolExec
        from draf.graph import Graph, Edge

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

        from draf.tool import Tool

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
        )
        with pytest.raises(RuntimeError, match="max_iterations"):
            await g.run(
                state={"input": "do it"}, tools=[AlwaysTool()], max_iterations=3
            )


class TestFlowReact:
    @pytest.mark.asyncio
    async def test_flow_react_direct_response(self, monkeypatch):
        from draf.flow import Flow

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

        flow = Flow("test")
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
        from draf.flow import Flow
        from draf.tool import Tool

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

        flow = Flow("test")
        flow.react(model="gpt-4", input_key="query", output_key="answer")
        g = flow.compile()
        r = await g.run(
            state={"query": "reverse it"}, tools=[ReverseTool()], max_iterations=10
        )
        assert r["answer"] == "reversed: olleh"
        assert len(r["messages"]) == 4

    @pytest.mark.asyncio
    async def test_flow_react_chains_after(self, monkeypatch):
        from draf.flow import Flow
        from draf.node import Transform

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

        flow = Flow("test")
        flow.react(model="gpt-4", input_key="query", output_key="answer")
        flow.step(
            Transform(
                {"action": "uppercase", "input_key": "answer", "output_key": "result"}
            )
        )
        g = flow.compile()
        r = await g.run(state={"query": "hi"}, max_iterations=5)
        assert r["result"] == "HELLO WORLD"
