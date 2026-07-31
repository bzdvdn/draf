"""ReAct agent: graph-visible tool-calling loop."""

import json
import os

import httpx

from draf.node.node import Node
from draf.node.llm import (
    LLM,
    _PROVIDER_DEFAULTS,
    _extract_message,
    _parse_text_tool_call,
)
from draf.tool.tool import coerce_args


class ReActAgent(Node):
    """Single-step LLM node for a graph-level ReAct loop.

    Executes one LLM call, then signals whether a tool was requested
    by setting ``state["_tool_call_name"]``.

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
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        model = cfg.get("model", "gpt-4")
        system = cfg.get("system", "")
        input_key = cfg.get("input_key", "input")
        output_key = cfg.get("output_key", "output")
        messages_key = cfg.get("messages_key", "messages")
        tool_call_key = cfg.get("tool_call_key", "_tool_call_name")
        temperature = cfg.get("temperature")
        max_tokens = cfg.get("max_tokens")
        response_format = cfg.get("response_format")
        provider_name = cfg.get("provider", "")
        base_url = cfg.get("base_url", "")
        api_key_env = cfg.get("api_key_env", "")
        chat_path = cfg.get("chat_path", "")
        auth_header = cfg.get("auth_header", "")
        auth_prefix = cfg.get("auth_prefix", "")

        messages = list(state.get(messages_key, []))
        if not messages:
            user_input = str(state.get(input_key, ""))
            if system:
                messages.append({"role": "system", "content": system})
            if user_input:
                messages.append({"role": "user", "content": user_input})

        tool_defs = []

        for t in ctx.tools.values():
            tool_defs.append(LLM._tool_to_schema(t))

        provider_key = (
            provider_name or LLM.DEFAULT_PROVIDER or model.split("-")[0].split("/")[0]
        )
        provider_key = provider_key.lower()
        defaults = _PROVIDER_DEFAULTS.get(provider_key, _PROVIDER_DEFAULTS["openai"])

        resolved_url = base_url or os.environ.get(
            f"{provider_key.upper()}_BASE_URL", defaults["base_url"]
        )
        resolved_env = api_key_env or defaults["api_key_env"]
        resolved_path = chat_path or defaults["chat_path"]

        api_key = ""
        if resolved_env:
            api_key = os.environ.get(resolved_env, "")
        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", "")

        headers = {"Content-Type": "application/json"}
        hdr_name = auth_header or defaults["auth_header"]
        if hdr_name and api_key:
            hdr_prefix = (
                auth_prefix if "auth_prefix" in cfg else defaults["auth_prefix"]
            )
            headers[hdr_name] = f"{hdr_prefix}{api_key}"

        body: dict = {"model": model, "messages": messages}
        if tool_defs:
            body["tools"] = tool_defs
        if response_format:
            body["response_format"] = response_format
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{resolved_url}{resolved_path}",
                headers=headers,
                json={**body, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()

        msg = _extract_message(data)

        result: dict = {}
        tool_calls = msg.get("tool_calls")

        if tool_calls:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            result[tool_call_key] = fn.get("name", "")
            result["_tool_call_id"] = tc.get("id", "")
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, dict):
                raw_args = json.dumps(raw_args)
            result["_tool_call_args"] = raw_args
            messages.append(msg)
        else:
            content = msg.get("content", "")
            parsed = _parse_text_tool_call(content) if tool_defs else None
            if parsed:
                name, args = parsed
                result[tool_call_key] = name
                result["_tool_call_id"] = f"call_{len(messages)}"
                result["_tool_call_args"] = json.dumps(args)
                messages.append(msg)
            else:
                result[output_key] = content
                result[tool_call_key] = ""
                messages.append({"role": "assistant", "content": content})

        result[messages_key] = messages
        return result


class ToolExec(Node):
    """Executes a tool signalled by ``ReActAgent`` and feeds the result
    back into the conversation history.

    Parameters:
        messages_key: State key for messages (default ``"messages"``).
        tool_call_key: Signal key (default ``"_tool_call_name"``).
    """

    type = "tool_exec"

    def __init__(
        self,
        config: dict | None = None,
        *,
        messages_key: str = "messages",
        tool_call_key: str = "_tool_call_name",
        **kwargs,
    ):
        merged = {
            "messages_key": messages_key,
            "tool_call_key": tool_call_key,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        messages_key = self.config.get("messages_key", "messages")
        tool_call_key = self.config.get("tool_call_key", "_tool_call_name")

        name = state.get(tool_call_key, "")
        args_raw = state.get("_tool_call_args", "{}")
        call_id = state.get("_tool_call_id", "")

        try:
            args = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            args = {}

        tool = ctx.tools.get(name)
        if tool:
            try:
                result = await tool.arun(**coerce_args(tool, args))
            except Exception as e:
                result = f"Error calling '{name}': {e}"
        else:
            result = f"Error: unknown tool '{name}'"

        messages = list(state.get(messages_key, []))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(result) if result is not None else "",
            }
        )

        return {
            messages_key: messages,
            tool_call_key: "",
            "_tool_call_args": "",
            "_tool_call_id": "",
        }
