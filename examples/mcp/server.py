"""A tiny MCP stdio server for the Draf MCP example.

Exposes two tools over stdio: ``add`` (integer arithmetic) and
``get_weather`` (a canned forecast).  It uses the low-level ``mcp`` SDK
and runs on the stdio transport, so it can be launched from
``run.py`` via ``mcp_tools(command=[...])``.

Usage:
    python examples/mcp/server.py   (usually launched by run.py)
"""

import asyncio

import mcp.types as types
from mcp.server import InitializationOptions
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

TOOLS = [
    types.Tool(
        name="add",
        description="Add two integers and return the sum",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    ),
    types.Tool(
        name="get_weather",
        description="Return the current weather for a city",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    ),
]

FORECAST = {
    "moscow": "snow, -12C",
    "london": "rain, +9C",
    "tokyo": "clear, +18C",
}


async def on_list_tools(ctx, params):
    return types.ListToolsResult(tools=TOOLS)


async def on_call_tool(ctx, params):
    args = params.arguments or {}
    if params.name == "add":
        total = int(args["a"]) + int(args["b"])
        text = str(total)
    elif params.name == "get_weather":
        text = FORECAST.get(str(args.get("city", "")).lower(), "unknown")
    else:
        raise RuntimeError(f"unknown tool {params.name}")
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)])


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
                capabilities=types.ServerCapabilities(tools=types.ToolsCapability()),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
