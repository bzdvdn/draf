"""Tool-approval resolution and parallel tool-call execution."""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections.abc import Mapping

from teff._async_util import gather_or_cancel
from teff.harness.formats import _tool_call_parts
from teff.tool.tool import Tool, coerce_args


async def resolve_approval(approver: typing.Any, name: str, args: dict) -> str:
    """Resolve a tool-approval decision for one tool call.

    *approver* may be:

    - ``"auto"`` (or ``None``) → ``"approve"``
    - ``"deny"`` → ``"deny"`` (no call ever runs)
    - ``"interactive"`` → prompt the operator on stdin
    - a callable ``(name, args) -> str | bool`` (sync or async)
      returning ``"approve"``/``"deny"``/``"pause"`` (or ``True``/``False``).

    Returns one of ``"approve"``, ``"deny"``, ``"pause"``.
    """
    if approver is None or approver == "auto":
        return "approve"
    if approver == "deny":
        return "deny"
    if approver == "interactive":
        import sys

        sys.stderr.write(
            f"\n[teff] approve tool call '{name}' with args {json.dumps(args)}? [y/N] "
        )
        sys.stderr.flush()
        answer = input().strip().lower()
        return "approve" if answer in ("y", "yes") else "deny"
    if callable(approver):
        result = approver(name, args)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, bool):
            return "approve" if result else "deny"
        return str(result).lower()
    return "approve"


def _tool_runtime_kwargs(tool: Tool, state: dict | None, ctx: typing.Any) -> dict:
    """Runtime kwargs (``__state__`` / ``__ctx__``) injected into a tool call.

    Only tools whose ``arun``/``run`` declare the matching parameter receive
    the value — plain domain tools are untouched.  ``__state__`` gives a
    tool read/write access to the enclosing workflow state (used by
    sub-agent tools that orchestrate their own LLM loop), and ``__ctx__``
    carries the :class:`~teff.node.context.ExecContext` for tracing.
    """
    method = tool.arun if type(tool).run is Tool.run else tool.run
    params = inspect.signature(method).parameters
    injected: dict = {}
    if state is not None and "__state__" in params:
        injected["__state__"] = state
    if ctx is not None and "__ctx__" in params:
        injected["__ctx__"] = ctx
    return injected


async def _run_one_tool_call(
    tc: dict,
    tools: Mapping[str, Tool],
    tool_error_mode: str,
    timeout: float | None = None,
    tool_retries: int = 0,
    approver: typing.Any = None,
    state: dict | None = None,
    ctx: typing.Any = None,
) -> str:
    name, raw_args, _call_id = _tool_call_parts(tc)
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return f"Error: invalid JSON arguments for tool '{name}'"
    if approver is not None and approver != "auto":
        decision = await resolve_approval(approver, name, args)
        if decision != "approve":
            return f"Tool call '{name}' was not approved ({decision})"
    tool = tools.get(name) if tools else None
    if tool is None:
        return f"Error: unknown tool '{name}'"

    attempts = tool_retries + 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            kwargs = coerce_args(tool, args)
            kwargs.update(_tool_runtime_kwargs(tool, state, ctx))
            coro = tool.arun(**kwargs)
            if timeout and timeout > 0:
                coro = asyncio.wait_for(coro, timeout=timeout)
            result = await coro
            return str(result) if result is not None else ""
        except asyncio.TimeoutError as exc:
            last_exc = exc
        except Exception as exc:  # noqa: BLE001 — mode drives final handling
            last_exc = exc
        if tool_error_mode == "raise" and attempt == attempts - 1:
            raise last_exc  # type: ignore[misc]
        if attempt == attempts - 1:
            if isinstance(last_exc, asyncio.TimeoutError):
                return f"Error calling '{name}': timed out after {timeout}s"
            return f"Error calling '{name}': {last_exc}"
    return f"Error calling '{name}': failed"


async def execute_tool_calls(
    tool_calls: list[dict],
    tools: Mapping[str, Tool],
    tool_error_mode: str = "message",
    timeout: float | None = None,
    tool_retries: int = 0,
    approver: typing.Any = None,
    state: dict | None = None,
    ctx: typing.Any = None,
) -> list[str]:
    """Execute *tool_calls* against *tools* in parallel.

    Each call resolves to a result string (errors become ``"Error ..."``
    messages unless *tool_error_mode* is ``"raise"``).  Each call is
    retried up to *tool_retries* times on failure and bounded by *timeout*
    seconds when set.  An optional *approver* gates each call before it
    runs (see :func:`resolve_approval`); non-``"approve"`` decisions
    short-circuit the call with a "not approved" message.

    *state* / *ctx* are injected into tools that declare ``__state__`` /
    ``__ctx__`` runtime kwargs (sub-agent tools), so they can read/write the
    enclosing workflow state and forward tracing.

    Args:
        tool_calls: List of tool-call dicts.
        tools: Tool registry (name -> ``Tool``).
        tool_error_mode: ``"message"`` or ``"raise"``.
        timeout: Per-tool timeout in seconds (``None`` = no limit).
        tool_retries: Extra attempts per tool call after a failure.
        approver: Approval policy (string or callable).
        state: Workflow state dict to expose to state-aware tools.
        ctx: :class:`~teff.node.context.ExecContext` to expose to tools.
    """
    if not tool_calls:
        return []
    return await gather_or_cancel(
        *(
            _run_one_tool_call(
                tc,
                tools,
                tool_error_mode,
                timeout,
                tool_retries,
                approver,
                state,
                ctx,
            )
            for tc in tool_calls
        )
    )
