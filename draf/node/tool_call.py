"""Tool-call node — invoke a registered tool deterministically.

Unlike ``react_agent``/``tool_exec`` (where the LLM decides which tool to
call), ``tool_call`` invokes one named tool with arguments taken straight
from the workflow config — great for deterministic side effects like
posting a GitLab note, approving a merge request, or sending a Telegram
message inside a ``workflow.yaml``.
"""

from __future__ import annotations

import typing

from draf.node.node import Node
from draf.prompt import render_template
from draf.tool.tool import coerce_args

if typing.TYPE_CHECKING:
    from draf.node.context import ExecContext


class ToolCall(Node):
    """Call a registered tool by name with config-driven arguments.

    Each argument value may contain ``{key}`` templates that are rendered
    from the current state before the call.  The tool's string result is
    written to ``output_key`` (default: ``output``).  When ``on_error`` is
    ``"message"`` a failure is stored under ``output_key`` as ``"error: ..."``
    instead of raising.

    Config:
        tool: Registered tool name to invoke.
        args: Mapping of tool argument name to value or ``{key}`` template.
        output_key: State key for the result (default ``"output"``).
        on_error: ``"raise"`` (default) or ``"message"``.
        max_chars: Truncate the result to this many characters.
    """

    type = "tool_call"

    def __init__(
        self,
        config: dict | None = None,
        *,
        tool: str = "",
        args: dict | None = None,
        output_key: str = "output",
        on_error: str = "raise",
        max_chars: int | None = None,
        **kwargs,
    ):
        merged = {
            "tool": tool,
            "args": args or {},
            "output_key": output_key,
            "on_error": on_error,
            "max_chars": max_chars,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx: "ExecContext", state: dict) -> dict:
        tool_name = str(self.config.get("tool", ""))
        if not tool_name:
            raise ValueError("tool_call requires 'tool'")
        tool = ctx.tool(tool_name)

        raw_args = dict(self.config.get("args") or {})
        args = {}
        for key, value in raw_args.items():
            if isinstance(value, str):
                value = render_template(value, state)
            args[key] = value
        args = coerce_args(tool, args)

        output_key = str(self.config.get("output_key", "output"))
        on_error = self.config.get("on_error", "raise")
        max_chars = self.config.get("max_chars")

        try:
            result = await tool.arun(**args)
        except Exception as exc:  # noqa: BLE001 — surfaced per on_error mode
            if on_error == "message":
                return {output_key: f"error: {exc}"}
            raise
        text = str(result)
        if max_chars is not None:
            text = text[: int(max_chars)]
        return {output_key: text}
