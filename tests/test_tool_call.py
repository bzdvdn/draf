import pytest

from teff.node import ExecContext
from teff.node.tool_call import ToolCall
from teff.tool import Tool


class EchoTool(Tool):
    """Deterministic tool for testing tool_call."""

    name = "echo"
    description = "Echo args back"

    async def arun(self, message: str = "", repeat: int = 1) -> str:  # type: ignore[override]
        return (message * repeat) or "empty"


class BoomTool(Tool):
    name = "boom"
    description = "Always raises"

    async def arun(self, **kwargs) -> str:
        raise RuntimeError("kaput")


def make_ctx(tools: list[Tool]) -> ExecContext:
    return ExecContext(state={}, tools={t.name: t for t in tools})


class TestToolCall:
    @pytest.mark.asyncio
    async def test_calls_tool_and_writes_output(self):
        ctx = make_ctx([EchoTool()])
        node = ToolCall(tool="echo", args={"message": "hi"}, output_key="result")
        out = await node.execute(ctx, {"message": "hi"})
        assert out == {"result": "hi"}

    @pytest.mark.asyncio
    async def test_renders_template_args_from_state(self):
        ctx = make_ctx([EchoTool()])
        node = ToolCall(tool="echo", args={"message": "{topic}"}, output_key="result")
        out = await node.execute(ctx, {"topic": "hello world"})
        assert out == {"result": "hello world"}

    @pytest.mark.asyncio
    async def test_coerces_args(self):
        ctx = make_ctx([EchoTool()])
        node = ToolCall(tool="echo", args={"message": "x", "repeat": "3"})
        out = await node.execute(ctx, {})
        assert out["output"] == "xxx"

    @pytest.mark.asyncio
    async def test_default_output_key(self):
        ctx = make_ctx([EchoTool()])
        node = ToolCall(tool="echo", args={"message": "y"})
        out = await node.execute(ctx, {})
        assert out == {"output": "y"}

    @pytest.mark.asyncio
    async def test_on_error_message(self):
        ctx = make_ctx([BoomTool()])
        node = ToolCall(tool="boom", on_error="message", output_key="result")
        out = await node.execute(ctx, {})
        assert out["result"].startswith("error:")

    @pytest.mark.asyncio
    async def test_on_error_raise_by_default(self):
        ctx = make_ctx([BoomTool()])
        node = ToolCall(tool="boom")
        with pytest.raises(RuntimeError, match="kaput"):
            await node.execute(ctx, {})

    @pytest.mark.asyncio
    async def test_max_chars_truncates(self):
        ctx = make_ctx([EchoTool()])
        node = ToolCall(
            tool="echo", args={"message": "abcdef"}, max_chars=3, output_key="result"
        )
        out = await node.execute(ctx, {})
        assert out == {"result": "abc"}

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        ctx = make_ctx([EchoTool()])
        node = ToolCall(tool="nope")
        with pytest.raises(KeyError, match="nope"):
            await node.execute(ctx, {})

    @pytest.mark.asyncio
    async def test_missing_tool_name_raises(self):
        ctx = make_ctx([EchoTool()])
        node = ToolCall(config={})
        with pytest.raises(ValueError, match="tool"):
            await node.execute(ctx, {})
