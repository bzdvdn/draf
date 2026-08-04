"""Bridge to Model Context Protocol (MCP) servers.

MCP is an open standard for exposing tools, resources and prompts to LLMs.
This module connects a Draf graph to an MCP server and exposes its tools as
regular :class:`~draf.tool.Tool` instances, so they work anywhere the built-in
tools do — LLM nodes, the ReAct agent, tool registries.

The ``mcp`` SDK ships with the core package; it is imported lazily so a plain
``import draf`` stays fast.

Usage::

    async with mcp_tools(url="http://localhost:8000/mcp") as tools:
        result = await graph.run(state, tools={t.name: t for t in tools})

    async with mcp_tools(command=["uvx", "mcp-server-git"]) as tools:
        result = await graph.run(state, tools={t.name: t for t in tools})
"""

from __future__ import annotations

import contextlib
import sys
import typing
from collections.abc import AsyncIterator
from functools import partial

from draf.tool.tool import Tool

if typing.TYPE_CHECKING:
    from mcp.client.session import ClientSession
    from mcp.types import Tool as McpToolSpec

__all__ = ["McpTool", "mcp_tools"]


def _format_mcp_content(result) -> str:
    """Flatten an MCP ``CallToolResult`` into a single string."""
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content.model_dump_json()
    parts: list[str] = []
    for block in getattr(result, "content", []):
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
        else:
            parts.append(block.model_dump_json())
    return "\n".join(parts)


class McpTool(Tool):
    """A :class:`~draf.tool.Tool` that forwards calls to an MCP server.

    Instances are created by :func:`mcp_tools`.  The tool's JSON schema
    comes from the server's tool definition instead of being inferred
    from type hints.
    """

    def __init__(self, session: "ClientSession", spec: "McpToolSpec"):
        super().__init__()
        self._session = session
        self.name = spec.name
        self.description = spec.description or ""
        # SDK stubs expose `inputSchema`, runtime uses `input_schema`.
        self.schema = spec.input_schema  # type: ignore[attr-defined]

    async def arun(self, **kwargs):
        result = await self._session.call_tool(self.name, kwargs)
        content = _format_mcp_content(result)
        if getattr(result, "is_error", False):
            raise RuntimeError(content or f"MCP tool '{self.name}' returned an error")
        return content


async def _list_all_tools(session: "ClientSession"):
    """List all tools from *session*, following pagination."""
    import mcp.types as types

    tools: list[McpToolSpec] = []
    cursor: str | None = None
    while True:
        params = (
            types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
        )
        result = await session.list_tools(params=params)
        tools.extend(result.tools)
        # SDK stubs omit `next_cursor`; runtime exposes it.
        cursor = result.next_cursor  # type: ignore[attr-defined]
        if not cursor:
            return tools


async def _connect_tools(read_stream, write_stream, client_info: dict | None = None):
    """Open a client session over *streams* and return ``(session, tools)``.

    The caller owns *session* and must close it (``await
    session.__aexit__(None, None, None)``) when done.
    """
    import mcp
    from mcp.client.session import ClientSession

    impl_name = "draf"
    impl_version = "unknown"
    try:
        import importlib.metadata

        impl_version = importlib.metadata.version("draf")
    except Exception:
        pass
    info = mcp.types.Implementation(name=impl_name, version=impl_version)
    if client_info:
        info = info.model_copy(update=client_info)

    session = ClientSession(read_stream, write_stream, client_info=info)
    await session.__aenter__()
    try:
        await session.initialize()
        tools = [McpTool(session, spec) for spec in await _list_all_tools(session)]
    except Exception:
        await session.__aexit__(*sys.exc_info())
        raise
    return session, tools


@contextlib.asynccontextmanager
async def mcp_tools(
    url: str | None = None,
    command: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    client_info: dict | None = None,
) -> AsyncIterator[list[McpTool]]:
    """Connect to an MCP server and yield its tools as Draf :class:`Tool`\\s.

    Exactly one of *url* or *command* must be given:

    - ``url``: Streamable HTTP endpoint of an MCP server, e.g.
      ``http://localhost:8000/mcp``.
    - ``command``: Subprocess invocation for a stdio server, e.g.
      ``["uvx", "mcp-server-git"]`` or
      ``["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]``.

    The session stays open for the duration of the ``async with`` block;
    tools keep working until it exits, after which the connection is closed.

    Args:
        url: Streamable HTTP endpoint.
        command: Stdio server command (list of argv tokens).
        env: Optional extra environment variables for stdio servers.
        cwd: Optional working directory for stdio servers.
        client_info: Optional dict overrides for the client
            ``Implementation`` advertised to the server.

    Yields:
        A list of :class:`McpTool` instances, one per server tool.
    """
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    if (url is None) == (command is None):
        raise ValueError("mcp_tools requires exactly one of 'url' or 'command'")

    stack = contextlib.AsyncExitStack()
    async with stack:
        if url is not None:
            # SDK stubs declare a wider tuple than runtime actually yields.
            read_stream, write_stream = await stack.enter_async_context(  # type: ignore[misc]
                streamable_http_client(url)
            )
        else:
            assert command is not None
            params = StdioServerParameters(
                command=command[0],
                args=command[1:],
                env=env,
                cwd=cwd,
            )
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(params)
            )

        session, tools = await _connect_tools(read_stream, write_stream, client_info)
        stack.push_async_callback(partial(session.__aexit__, None, None, None))
        yield tools
