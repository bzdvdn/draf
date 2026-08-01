"""ReAct agent: graph-visible tool-calling loop."""

import json

from draf.harness import (
    Harness,
    execute_tool_calls,
    parse_text_tool_call,
    tool_to_schema,
)
from draf.node.node import Node
from draf.node.llm import LLM
from draf.skill import resolve_skills, scope_tools, skills_instructions


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
    """

    type = "react_agent"

    def __init__(
        self,
        config: dict | None = None,
        *,
        model: str = "gpt-4",
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
        use_tools: bool | list[str] = True,
        skills: list | None = None,
        skill_dir: str = "skills",
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
        if not messages:
            user_input = str(state.get(input_key, ""))
            if system:
                messages.append({"role": "system", "content": system})
            if user_input:
                messages.append({"role": "user", "content": user_input})

        tool_defs = [
            tool_to_schema(t) for t in scope_tools(ctx.tools, cfg, skills).values()
        ]

        harness = Harness.from_config(cfg, default_provider=LLM.DEFAULT_PROVIDER)
        tracer = getattr(ctx, "tracer", None)
        if tracer is not None:
            async def on_llm(provider, model, prompt, completion, duration):
                tracer.llm(provider, model, prompt, completion, duration)

            harness.on_llm = on_llm

        reply = await harness.call(messages, tools=tool_defs or None)

        result: dict = {}
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
            parsed = parse_text_tool_call(content) if tool_defs else None
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
    """

    type = "tool_exec"

    def __init__(
        self,
        config: dict | None = None,
        *,
        messages_key: str = "messages",
        tool_call_key: str = "_tool_call_name",
        tool_error_mode: str = "message",
        use_tools: bool | list[str] = True,
        skills: list | None = None,
        skill_dir: str = "skills",
        **kwargs,
    ):
        merged = {
            "messages_key": messages_key,
            "tool_call_key": tool_call_key,
            "tool_error_mode": tool_error_mode,
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
        results = await execute_tool_calls(calls, scoped, tool_error_mode)

        messages = list(state.get(messages_key, []))
        for call, res in zip(calls, results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": str(res) if res is not None else "",
                }
            )

        return {
            messages_key: messages,
            tool_call_key: "",
            "_tool_calls": [],
            "_tool_call_args": "",
            "_tool_call_id": "",
        }
