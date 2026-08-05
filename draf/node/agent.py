"""ReAct agent: graph-visible tool-calling loop."""

import asyncio
import json
import typing

from draf.harness import (
    Harness,
    execute_tool_calls,
    parse_text_tool_call,
    resolve_approval,
    tool_to_schema,
)
from draf.node.interrupt import GraphInterrupt
from draf.node.node import Node
from draf.skill import resolve_skills, scope_tools, skills_instructions
from draf.state import reducer_appends
from draf.stream import StreamEvent


class ReActAgent(Node):
    """Single-step LLM node for a graph-level ReAct loop.

    Executes one LLM call, then signals any requested tools by setting
    ``state["_tool_calls"]`` (a list of ``{id, name, args}``) and a
    non-empty ``state["_tool_call_name"]``.

    When the LLM responds without calling a tool, the output key is
    set and ``_tool_call_name`` is cleared — the parent graph stops
    because no outgoing edge matches.

    Expected graph edges::

        agent  ──(_tool_call_name!=)──→  tool_exec
        tool_exec  ──(unconditional)──→  agent

    Parameters:
        model: Model name (e.g. ``gpt-4``).
        system: System prompt.
        input_key: State key for user input (default ``"input"``).
        output_key: State key for final response (default ``"output"``).
        messages_key: State key for conversation (default ``"messages"``).
        tool_call_key: Signal key (default ``"_tool_call_name"``).
        temperature: Sampling temperature.
        max_tokens: Max tokens in response.
        response_format: ``{"type": "json_object"}`` etc.
        provider: Force a provider (auto-detected from model).
        base_url: Custom base URL.
        api_key_env: Custom env var name for API key.
        chat_path: Custom API path.
        auth_header: Custom auth header name.
        auth_prefix: Custom auth header prefix.
        max_tool_rounds: Round limit used by the harness loop.
        parse_text_tool_calls: Decode text-embedded tool calls.
        tool_error_mode: ``"message"`` (default) or ``"raise"``.
        tool_timeout: Per-tool execution timeout in seconds.
        tool_retries: Extra attempts per tool call after a failure.
        max_retries: HTTP request retries (429/5xx/timeouts).
        fallbacks: Fallback model names for provider failover.
        tool_approval: Gate on tool execution — ``"auto"`` (default),
            ``"deny"``, ``"interactive"`` (ask on stdin), or a callable
            ``(name, args) -> bool | str``.  ``"pause"`` decisions pause
            the run as a :class:`~draf.node.interrupt.GraphInterrupt`.
        stream: Stream the final assistant text (tokens forwarded to
            ``on_token`` and stream events).
        on_token: Callback ``(token: str) -> None`` for streaming.
    """

    type = "react_agent"

    def __init__(
        self,
        config: dict | None = None,
        *,
        model: str | None = None,
        system: str = "",
        input_key: str = "input",
        output_key: str = "output",
        messages_key: str = "messages",
        tool_call_key: str = "_tool_call_name",
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        chat_path: str | None = None,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
        use_tools: str | list[str] | None = None,
        skills: list | None = None,
        skill_dir: str = "skills",
        max_tool_rounds: int | None = None,
        parse_text_tool_calls: bool | None = None,
        tool_error_mode: str | None = None,
        tool_timeout: float | None = None,
        tool_retries: int = 0,
        max_retries: int = 2,
        fallbacks: list[str] | None = None,
        tool_approval: typing.Any = None,
        stream: bool = False,
        on_token: typing.Callable[[str], None] | None = None,
        **kwargs,
    ):
        merged = {
            "model": model,
            "system": system,
            "input_key": input_key,
            "output_key": output_key,
            "messages_key": messages_key,
            "tool_call_key": tool_call_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "provider": provider,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "chat_path": chat_path,
            "auth_header": auth_header,
            "auth_prefix": auth_prefix,
            "use_tools": use_tools,
            "skills": skills,
            "skill_dir": skill_dir,
            "max_tool_rounds": max_tool_rounds,
            "parse_text_tool_calls": parse_text_tool_calls,
            "tool_error_mode": tool_error_mode,
            "tool_timeout": tool_timeout,
            "tool_retries": tool_retries,
            "max_retries": max_retries,
            "fallbacks": fallbacks,
            "tool_approval": tool_approval,
            "stream": stream,
            "on_token": on_token,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        system = cfg.get("system", "")
        input_key = cfg.get("input_key", "input")
        output_key = cfg.get("output_key", "output")
        messages_key = cfg.get("messages_key", "messages")
        tool_call_key = cfg.get("tool_call_key", "_tool_call_name")

        skills = resolve_skills(cfg)
        skill_text = skills_instructions(skills)
        if skill_text:
            system = f"{system}\n\n{skill_text}" if system else skill_text

        messages = list(state.get(messages_key, []))
        start = len(messages)
        if not messages:
            user_input = str(state.get(input_key, ""))
            if system:
                messages.append({"role": "system", "content": system})
            if user_input:
                messages.append({"role": "user", "content": user_input})

        tool_defs = [
            tool_to_schema(t) for t in scope_tools(ctx.tools, cfg, skills).values()
        ]

        harness = Harness.from_config(
            cfg,
            default_provider=getattr(ctx, "default_provider", None),
            default_model=getattr(ctx, "default_model", None),
            providers=getattr(ctx, "providers", None),
        )
        tracer = getattr(ctx, "tracer", None)
        if tracer is not None:

            async def on_llm(provider, model, prompt, completion, duration):
                tracer.llm(provider, model, prompt, completion, duration)

            harness.on_llm = on_llm

        emit = getattr(ctx, "emit", None)
        on_token_cfg = cfg.get("on_token")

        async def token_sink(token: str) -> None:
            if on_token_cfg is not None:
                res = on_token_cfg(token)
                if asyncio.iscoroutine(res):
                    await res
            if emit is not None:
                await emit(
                    StreamEvent(
                        "token",
                        node_id=ctx.node_id,
                        node_type=ctx.node_type,
                        data={
                            "token": token,
                            "provider": harness.provider_key,
                            "model": str(cfg.get("model", "")),
                        },
                    )
                )

        want_stream = bool(cfg.get("stream", False))
        if want_stream and not tool_defs:
            harness.on_token = token_sink

        reply = await harness.call(
            messages,
            tools=tool_defs or None,
            stream=want_stream and not tool_defs,
        )

        result: dict = {}

        # The graph loop (agent → tool → agent) is what repeats, so the node
        # itself must track how many times it has been visited.  Once the
        # round budget is spent we stop signalling tools even if the model
        # keeps asking, letting the loop end on this node.
        round_key = f"_react_round_{ctx.node_id}"
        round_count = int(state.get(round_key, 0)) + 1
        result[round_key] = round_count
        max_rounds = cfg.get("max_tool_rounds")
        budget_spent = max_rounds is not None and round_count > max_rounds

        if budget_spent:
            content = reply.content or ""
            messages.append({"role": "assistant", "content": content})
            result[output_key] = content
            result[tool_call_key] = ""
            result["_tool_calls"] = []
        else:
            tool_calls = reply.message.get("tool_calls")

            if tool_calls:
                calls: list[dict] = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    raw = fn.get("arguments", "{}")
                    if isinstance(raw, dict):
                        raw = json.dumps(raw)
                    calls.append(
                        {
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "args": raw,
                        }
                    )
                result[tool_call_key] = "pending"
                result["_tool_calls"] = calls
                messages.append(reply.message)
            else:
                content = reply.content
                parse_cfg = cfg.get("parse_text_tool_calls", True)
                if parse_cfg is None:
                    parse_cfg = True
                parsed = (
                    parse_text_tool_call(content) if tool_defs and parse_cfg else None
                )
                if parsed:
                    name, args = parsed
                    result[tool_call_key] = "pending"
                    result["_tool_calls"] = [
                        {
                            "id": f"call_{len(messages)}",
                            "name": name,
                            "args": json.dumps(args),
                        }
                    ]
                    messages.append({"role": "assistant", "content": content})
                else:
                    result[output_key] = content
                    result[tool_call_key] = ""
                    result["_tool_calls"] = []
                    messages.append({"role": "assistant", "content": content})

        if reducer_appends((ctx.reducers or {}).get(messages_key)):
            result[messages_key] = messages[start:]
        else:
            result[messages_key] = messages
        return result


class ToolExec(Node):
    """Executes tools signalled by :class:`ReActAgent` in parallel and feeds
    the results back into the conversation history.

    Handles multiple tool calls per round: the agent writes the whole
    ``_tool_calls`` list, which is executed concurrently and appended as
    ``tool`` messages in one go.  Falls back to the legacy single-call
    signals (``_tool_call_name`` / ``_tool_call_args`` / ``_tool_call_id``).

    Parameters:
        messages_key: State key for messages (default ``"messages"``).
        tool_call_key: Signal key (default ``"_tool_call_name"``).
        tool_error_mode: ``"message"`` (default) or ``"raise"`` — when
            ``"raise"``, a tool failure propagates to the graph error path
            (e.g. an ``__error__`` edge) instead of becoming a tool message.
        tool_timeout: Per-tool execution timeout in seconds.
        tool_retries: Extra attempts per tool call after a failure.
        tool_approval: Gate on tool execution — ``"auto"`` (default),
            ``"deny"``, ``"interactive"`` (ask on stdin), or a callable
            ``(name, args) -> bool | str`` (sync or async).  A ``"pause"``
            decision pauses the run as a :class:`GraphInterrupt`; ``"deny"``
            short-circuits the call with a "denied" tool message.
    """

    type = "tool_exec"

    def __init__(
        self,
        config: dict | None = None,
        *,
        messages_key: str = "messages",
        tool_call_key: str = "_tool_call_name",
        tool_error_mode: str = "message",
        tool_timeout: float | None = None,
        tool_retries: int = 0,
        tool_approval: typing.Any = None,
        use_tools: str | list[str] | None = None,
        skills: list | None = None,
        skill_dir: str = "skills",
        **kwargs,
    ):
        merged = {
            "messages_key": messages_key,
            "tool_call_key": tool_call_key,
            "tool_error_mode": tool_error_mode,
            "tool_timeout": tool_timeout,
            "tool_retries": tool_retries,
            "tool_approval": tool_approval,
            "use_tools": use_tools,
            "skills": skills,
            "skill_dir": skill_dir,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        messages_key = self.config.get("messages_key", "messages")
        tool_call_key = self.config.get("tool_call_key", "_tool_call_name")
        tool_error_mode = self.config.get("tool_error_mode", "message")
        tool_timeout = self.config.get("tool_timeout")
        tool_retries = int(self.config.get("tool_retries", 0))
        approver = self.config.get("tool_approval")

        calls = list(state.get("_tool_calls") or [])
        if not calls and state.get(tool_call_key):
            calls = [
                {
                    "id": state.get("_tool_call_id", ""),
                    "name": state.get(tool_call_key, ""),
                    "args": state.get("_tool_call_args", "{}"),
                }
            ]

        skills = resolve_skills(self.config)
        scoped = scope_tools(ctx.tools, self.config, skills)

        # After a pause/interrupt, the operator's decision comes back in the
        # resume payload under the interrupt key; use it instead of re-asking.
        resumed = state.get("tool_approval")
        resumed = resumed if resumed in ("approve", "deny") else None

        to_run = calls
        denied: list[tuple[str, str, str]] = []
        if approver is not None and approver != "auto" and calls:
            to_run = []
            for call in calls:
                name = call.get("name", "")
                try:
                    args = (
                        json.loads(call.get("args", "{}")) if call.get("args") else {}
                    )
                except json.JSONDecodeError:
                    args = {}
                if resumed is not None:
                    decision = resumed
                else:
                    decision = await resolve_approval(approver, name, args)
                if decision == "pause":
                    raise GraphInterrupt(
                        key="tool_approval",
                        prompt=(
                            f"Approve tool call '{name}' with args {json.dumps(args)}?"
                        ),
                    )
                if decision != "approve":
                    denied.append((name, call.get("id", ""), decision))
                else:
                    to_run.append(call)

        results = await execute_tool_calls(
            to_run, scoped, tool_error_mode, tool_timeout, tool_retries
        )

        messages = list(state.get(messages_key, []))
        start = len(messages)
        for call, res in zip(to_run, results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": str(res) if res is not None else "",
                }
            )
        for name, call_id, decision in denied:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"Tool call '{name}' was not approved ({decision})",
                }
            )

        out: dict = {
            tool_call_key: "",
            "_tool_calls": [],
            "_tool_call_args": "",
            "_tool_call_id": "",
        }
        if reducer_appends((ctx.reducers or {}).get(messages_key)):
            out[messages_key] = messages[start:]
        else:
            out[messages_key] = messages
        return out
