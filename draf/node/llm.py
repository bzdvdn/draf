"""LLM chat node — multi-provider, tool calling, structured output."""

import os
import json
import re
import time
import inspect
import typing

import httpx

from draf.node.node import Node
from draf.prompt import render_template
from draf.tool.tool import Tool, coerce_args

_PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "OPENAI_API_KEY",
        "chat_path": "/chat/completions",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "api_key_env": "ANTHROPIC_API_KEY",
        "chat_path": "/messages",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "DEEPSEEK_API_KEY",
        "chat_path": "/chat/completions",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "auth_header": "",
        "auth_prefix": "",
        "api_key_env": "",
        "chat_path": "/api/chat",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "MISTRAL_API_KEY",
        "chat_path": "/chat/completions",
    },
}


def _extract_message(data: dict) -> dict:
    """Normalise response formats to ``{role, content, tool_calls}``.

    Handles OpenAI (``data["choices"][0]["message"]``) and
    Ollama (``data["message"]`` at root).
    """
    msg = data.get("choices", [{}])[0].get("message", {})
    if not msg and "message" in data:
        msg = data["message"]
    return msg


def _extract_json_object(text: str, start: int) -> str | None:
    """Return the balanced JSON object starting at ``text[start] == '{'``."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_text_tool_call(content: str) -> tuple[str, dict] | None:
    """Parse a tool call embedded in plain text content.

    Local models sometimes emit ``{"name": "rag", "parameters": {...}}``
    or ``{"name": "rag", "arguments": {...}}`` as text instead of using
    the structured ``tool_calls`` field. Returns ``(name, args)`` if found.
    """
    m = re.search(r'"name"\s*:\s*"([^"]+)"', content)
    if not m:
        return None
    name = m.group(1)
    for key in ("parameters", "arguments"):
        idx = content.find(f'"{key}"')
        if idx == -1:
            continue
        brace = content.find("{", content.find(":", idx))
        if brace == -1:
            continue
        obj = _extract_json_object(content, brace)
        if obj is None:
            continue
        try:
            args = json.loads(obj)
        except json.JSONDecodeError:
            args = {}
        return name, args
    return name, {}


_JSON_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _py_type_to_json(tp: type) -> str:
    return _JSON_TYPE_MAP.get(tp, "string")


def _extract_usage(data: dict) -> tuple[int, int]:
    """Extract ``(prompt_tokens, completion_tokens)`` from an LLM response.

    Handles both OpenAI-style (``data["usage"]``) and Ollama-style
    (``data["prompt_eval_count"]`` / ``data["eval_count"]``) formats.
    """
    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None:
        prompt = data.get("prompt_eval_count", 0)
    if completion is None:
        completion = data.get("eval_count", 0)
    return int(prompt or 0), int(completion or 0)


class LLM(Node):
    """Call an LLM chat API with tool calling and structured output.

    Parameters:
        model: Model name (e.g. ``gpt-4``, ``llama3.1:8b``).
        system: System prompt.  Supports ``{key}`` placeholders rendered
            from state (see :func:`draf.prompt.render_template`).
        prompt: User prompt template.  Supports ``{key}`` placeholders
            rendered from state, e.g. ``"составь план для ремонта {type} "
            "на сумму {summ}"``.  Overrides *input_key* when set.
        input_key: State key for user message (default: whole state).
        output_key: State key for the response (default ``"output"``).
        provider: Provider name (``"openai"``, ``"ollama"``, etc.).
            Falls back to ``LLM.DEFAULT_PROVIDER``, then auto-detection
            from model name.
        use_tools: If ``True``, auto-generate tool definitions from
            ``ctx.tools``.
        temperature: Sampling temperature.
        max_tokens: Max tokens in response.
        response_format: ``{"type": "json_object"}`` etc.
        stream: If ``True``, use SSE streaming.
            Automatically disabled when tool calling is active.
        on_token: Optional callback ``(token: str) -> None`` for streaming.
        base_url: Custom base URL (overrides provider default).
        chat_path: Custom API path (overrides provider default).
        auth_header: Custom auth header name.
        auth_prefix: Custom auth header prefix.
        api_key_env: Custom env var for API key.
        tools: List of raw tool definition dicts.
        messages_key: State key for message history.
            If set, the conversation history is read/written from/to
            ``state[messages_key]`` instead of being built fresh each call.
        response_path: Dot-separated path to extract content from response.
    """

    type = "llm_chat"
    _MAX_TOOL_ROUNDS = 10
    DEFAULT_PROVIDER: str | None = None

    def __init__(
        self,
        config: dict | None = None,
        *,
        model: str = "gpt-4",
        system: str = "",
        prompt: str | None = None,
        input_key: str | None = None,
        output_key: str = "output",
        provider: str | None = None,
        use_tools: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        stream: bool = False,
        on_token: typing.Callable[[str], None] | None = None,
        base_url: str | None = None,
        chat_path: str | None = None,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
        api_key_env: str | None = None,
        tools: list[dict] | None = None,
        messages_key: str | None = None,
        response_path: str = "",
        **kwargs: typing.Any,
    ):
        merged = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "input_key": input_key,
            "output_key": output_key,
            "provider": provider,
            "use_tools": use_tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "stream": stream,
            "on_token": on_token,
            "base_url": base_url,
            "chat_path": chat_path,
            "auth_header": auth_header,
            "auth_prefix": auth_prefix,
            "api_key_env": api_key_env,
            "messages_key": messages_key,
            "response_path": response_path,
            **(config or {}),
            **kwargs,
        }
        # ensure tools is always a list
        merged.setdefault("tools", tools or [])
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        provider_key = self._resolve_provider(cfg)
        defaults = _PROVIDER_DEFAULTS.get(provider_key, _PROVIDER_DEFAULTS["openai"])

        base_url = cfg.get("base_url") or ""
        chat_path = cfg.get("chat_path") or ""
        api_key_env = cfg.get("api_key_env") or ""

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
        hdr_name = cfg.get("auth_header") or defaults["auth_header"]
        if hdr_name and api_key:
            hdr_prefix = cfg.get("auth_prefix") or defaults["auth_prefix"]
            headers[hdr_name] = f"{hdr_prefix}{api_key}"

        has_messages_key = cfg.get("messages_key") and state.get(cfg["messages_key"])
        if has_messages_key:
            messages = list(state[cfg["messages_key"]])
        else:
            prompt = cfg.get("prompt")
            input_key = cfg.get("input_key")
            if prompt:
                user_message = render_template(prompt, state)
            elif input_key:
                user_message = str(state.get(input_key, ""))
            else:
                user_message = str(state)
            messages = []
            system = cfg.get("system", "")
            if system:
                messages.append(
                    {"role": "system", "content": render_template(system, state)}
                )
            if user_message:
                messages.append({"role": "user", "content": user_message})

        tool_defs: list[dict] = list(cfg.get("tools", []))
        if cfg.get("use_tools", False):
            for t in ctx.tools.values():
                tool_defs.append(self._tool_to_schema(t))

        body: dict = {"model": cfg["model"], "messages": messages}
        if tool_defs:
            body["tools"] = tool_defs
        if cfg.get("response_format"):
            body["response_format"] = cfg["response_format"]
        if cfg.get("temperature") is not None:
            body["temperature"] = cfg["temperature"]
        if cfg.get("max_tokens") is not None:
            body["max_tokens"] = cfg["max_tokens"]

        has_tools = bool(tool_defs)
        content = ""
        output_key = cfg.get("output_key", "output")

        if cfg.get("stream", False) and not has_tools:
            content = await self._post_llm_stream(
                resolved_url, resolved_path, headers, body, cfg.get("on_token")
            )
        else:
            for _round in range(self._MAX_TOOL_ROUNDS if has_tools else 1):
                t0 = time.monotonic()
                data = await self._post_llm(resolved_url, resolved_path, headers, body)
                self._record_llm(ctx, cfg, provider_key, data, t0)
                msg = _extract_message(data)
                content = self._extract_content(
                    data,
                    provider_key,
                    cfg.get("response_path", ""),
                    msg.get("content", ""),
                )
                tool_calls = msg.get("tool_calls")

                if has_tools and not tool_calls:
                    parsed = _parse_text_tool_call(content)
                    if parsed:
                        name, args = parsed
                        call_id = f"call_{len(body['messages'])}"
                        tool_calls = [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ]
                        msg = {**msg, "tool_calls": tool_calls}

                if not has_tools or not tool_calls:
                    break

                body["messages"].append(msg)
                for tc in tool_calls:
                    tc_result = await self._execute_tool_call(tc, ctx)
                    body["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tc_result,
                        }
                    )

        return {output_key: content or ""}

    def _resolve_provider(self, cfg: dict) -> str:
        p = cfg.get("provider") or self.DEFAULT_PROVIDER
        if p:
            return p.lower()
        detected = cfg.get("model", "gpt-4").split("-")[0].split("/")[0]
        return detected.lower()

    def _record_llm(
        self, ctx, cfg: dict, provider_key: str, data: dict, t0: float
    ) -> None:
        """Record an LLM call's token usage on the context tracer."""
        tracer = getattr(ctx, "tracer", None)
        if tracer is None:
            return
        prompt, completion = _extract_usage(data)
        from draf.trace import _ms

        tracer.llm(
            provider_key,
            str(cfg.get("model", "")),
            prompt,
            completion,
            _ms(t0),
        )

    async def _post_llm(
        self, base_url: str, path: str, headers: dict, body: dict
    ) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{base_url}{path}", headers=headers, json={**body, "stream": False}
            )
            response.raise_for_status()
            return response.json()

    async def _post_llm_stream(
        self,
        base_url: str,
        path: str,
        headers: dict,
        body: dict,
        on_token: typing.Callable[[str], None] | None = None,
    ) -> str:
        body = {**body, "stream": True}
        content = ""
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{base_url}{path}", headers=headers, json=body
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        content += token
                        if on_token:
                            on_token(token)
        return content

    async def _execute_tool_call(self, tc: dict, ctx) -> str:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments for tool '{name}'"

        tool = ctx.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"

        try:
            result = await tool.arun(**coerce_args(tool, args))
        except Exception as e:
            return f"Error calling '{name}': {e}"
        return str(result) if result is not None else ""

    def _extract_content(
        self, data: dict, provider: str, path: str = "", fallback: str = ""
    ) -> str:
        if path:
            parts = path.split(".")
            val = data
            for p in parts:
                if p.isdigit():
                    val = val[int(p)]
                else:
                    val = val.get(p, "")
            return str(val) if val else ""

        if provider == "anthropic":
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block.get("text", "")
            return ""

        if provider == "ollama":
            return data.get("message", {}).get("content", "")

        return fallback

    @staticmethod
    def _tool_to_schema(tool: Tool) -> dict:
        run_method = tool.run
        if type(tool).run is Tool.run and type(tool).arun is not Tool.arun:
            run_method = tool.arun
        sig = inspect.signature(run_method)
        try:
            hints = typing.get_type_hints(run_method)
        except Exception:
            hints = {}

        properties: dict = {}
        required: list[str] = []

        for pname, param in sig.parameters.items():
            if pname in ("self", "kwargs", "args"):
                continue
            json_type = _py_type_to_json(hints.get(pname, str))
            prop: dict = {"type": json_type}
            if param.default is not inspect.Parameter.empty:
                if param.default is not None:
                    prop["default"] = param.default
            else:
                required.append(pname)
            properties[pname] = prop

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
