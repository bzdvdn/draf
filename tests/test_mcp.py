"""Tests for the MCP (Model Context Protocol) tool bridge."""

import asyncio
import contextlib
import sys
import textwrap

import pytest

from teff.node.llm import LLM
from teff.tool import McpTool, mcp_tools
from teff.tool.mcp import _connect_tools

pytest.importorskip("mcp")

import mcp.types as types  # noqa: E402
from mcp.server import InitializationOptions  # noqa: E402
from mcp.server.lowlevel import Server  # noqa: E402
from mcp.shared.memory import create_client_server_memory_streams  # noqa: E402

TOOLS = [
    types.Tool(
        name="echo",
        description="Echo the text back",
        input_schema={  # type: ignore[call-arg]
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    ),
    types.Tool(
        name="add",
        description="Add two integers",
        input_schema={  # type: ignore[call-arg]
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    ),
    types.Tool(
        name="boom",
        description="Always fails",
        input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
    ),
]


async def _server_ctx(tools=TOOLS, calls=None):
    """Start a low-level MCP server over in-memory streams."""

    async def on_list_tools(ctx, params):
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx, params):
        if calls is not None:
            calls.append((params.name, params.arguments or {}))
        if params.name == "echo":
            text = (params.arguments or {}).get("text", "")
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)]
            )
        if params.name == "add":
            args = params.arguments or {}
            total = int(args["a"]) + int(args["b"])
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(total))]
            )
        if params.name == "boom":
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="kaboom")],
                is_error=True,
            )
        raise RuntimeError(f"unknown tool '{params.name}'")

    server = Server(
        "test-server",
        version="1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )

    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        read_stream, write_stream = server_streams
        task = asyncio.create_task(
            server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="test-server",
                    server_version="1.0",
                    capabilities=types.ServerCapabilities(
                        tools=types.ToolsCapability()
                    ),
                ),
            )
        )
        try:
            yield client_streams
        finally:
            task.cancel()


running_server = contextlib.asynccontextmanager(_server_ctx)


async def connect_tools(client_streams):
    read_stream, write_stream = client_streams
    session, tools = await _connect_tools(read_stream, write_stream)
    return session, tools


async def test_connect_lists_tools():
    async with running_server() as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            assert [t.name for t in tools] == ["echo", "add", "boom"]
            assert tools[0].description == "Echo the text back"
            assert tools[1].schema == TOOLS[1].input_schema
            assert all(isinstance(t, McpTool) for t in tools)
        finally:
            await session.__aexit__(None, None, None)


async def test_arun_forwards_args_and_returns_text():
    calls = []
    async with running_server(calls=calls) as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            assert await tools[0].arun(text="hello world") == "hello world"
            assert await tools[1].arun(a=2, b=3) == "5"
            assert calls == [
                ("echo", {"text": "hello world"}),
                ("add", {"a": 2, "b": 3}),
            ]
        finally:
            await session.__aexit__(None, None, None)


async def test_arun_raises_on_error_result():
    async with running_server() as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            boom = [t for t in tools if t.name == "boom"][0]
            with pytest.raises(RuntimeError, match="kaboom"):
                await boom.arun()
        finally:
            await session.__aexit__(None, None, None)


async def test_arun_raises_on_unknown_tool():
    async with running_server() as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            with pytest.raises(Exception, match="unknown tool"):
                await session.call_tool("nonexistent", {})
        finally:
            await session.__aexit__(None, None, None)


async def test_tool_schema_used_by_llm():
    async with running_server() as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            schema = LLM._tool_to_schema(tools[0])
            fn = schema["function"]
            assert fn["name"] == "echo"
            assert fn["parameters"] == TOOLS[0].input_schema
        finally:
            await session.__aexit__(None, None, None)


async def test_mcp_tools_requires_exactly_one_transport():
    with pytest.raises(ValueError, match="exactly one"):
        async with mcp_tools():
            pass
    with pytest.raises(ValueError, match="exactly one"):
        async with mcp_tools(url="http://x", command=["y"]):
            pass


SERVER_SCRIPT = textwrap.dedent(
    """
    import asyncio

    import mcp.types as types
    from mcp.server import InitializationOptions
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server


    async def on_list_tools(ctx, params):
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="add",
                    description="Add two integers",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                )
            ]
        )


    async def on_call_tool(ctx, params):
        if params.name == "add":
            args = params.arguments or {}
            total = int(args["a"]) + int(args["b"])
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(total))]
            )
        raise RuntimeError(f"unknown tool {params.name}")


    async def main():
        server = Server(
            "demo-server",
            version="1.0",
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
        )
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="demo-server",
                    server_version="1.0",
                    capabilities=types.ServerCapabilities(
                        tools=types.ToolsCapability()
                    ),
                ),
            )


    if __name__ == "__main__":
        asyncio.run(main())
    """
)


async def test_stdio_end_to_end(tmp_path):
    script = tmp_path / "mcp_server.py"
    script.write_text(SERVER_SCRIPT)

    async with mcp_tools(command=[sys.executable, str(script)]) as tools:
        assert [t.name for t in tools] == ["add"]
        assert tools[0].description == "Add two integers"
        assert tools[0].schema["required"] == ["a", "b"]
        assert await tools[0].arun(a=20, b=22) == "42"
