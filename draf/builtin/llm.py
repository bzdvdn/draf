"""LLM chat node — multi-provider, tool calling, structured output."""

import os
import json
import inspect
import typing

import httpx

from draf.node.node import Node
from draf.tool.tool import Tool

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


class LLM(Node):
    """Call an LLM chat API with tool calling and structured output.

    Config options (all optional, defaults vary by provider):

    - ``model`` — Model name (e.g. ``gpt-4``).
    - ``system`` — System prompt.
    - ``input_key`` — State key for user message (default: whole state).
    - ``output_key`` — State key for the response (default: ``output``).
    - ``provider`` — Force a provider (auto-detected from model).
    - ``api_key_env`` — Custom env var for the API key.
    - ``auth_header`` / ``auth_prefix`` — Custom auth header.
    - ``base_url`` / ``chat_path`` — Custom endpoint.
    - ``tools`` — List of raw tool definitions (dicts) for function calling.
    - ``use_tools`` — If ``True``, auto-generate tool definitions from
      ``ctx.tools`` (``ExecutorContext`` tool registry).
    - ``response_format`` — ``{"type": "json_object"}`` or
      ``{"type": "json_schema", "json_schema": {...}}`` for structured output.
    - ``response_path`` — Dot-separated path to extract content.
    - ``temperature`` — Sampling temperature (default: unset).
    - ``max_tokens`` — Max tokens in response (default: unset).
    - ``stream`` — If ``True``, use SSE streaming (default: ``False``).
      Automatically disabled when tool calling is active.
    - ``on_token`` — Optional callback ``(token: str) -> None`` called
      for each token during streaming.
    """

    type = "llm_chat"
    _MAX_TOOL_ROUNDS = 10

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        model = cfg.get("model", "gpt-4")
        system = cfg.get("system", "")
        input_key = cfg.get("input_key")
        output_key = cfg.get("output_key", "output")
        tools_cfg = cfg.get("tools", [])
        use_tools = cfg.get("use_tools", False)
        stream = cfg.get("stream", False)
        on_token = cfg.get("on_token")
        response_format = cfg.get("response_format")
        temperature = cfg.get("temperature")
        max_tokens = cfg.get("max_tokens")
        provider_name = cfg.get("provider", "")
        api_key_env = cfg.get("api_key_env", "")
        auth_header = cfg.get("auth_header", "")
        auth_prefix = cfg.get("auth_prefix", "")
        base_url = cfg.get("base_url", "")
        chat_path = cfg.get("chat_path", "")
        response_path = cfg.get("response_path", "")

        provider_key = provider_name or model.split("-")[0].split("/")[0]
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
            hdr_prefix = auth_prefix if "auth_prefix" in cfg else defaults["auth_prefix"]
            headers[hdr_name] = f"{hdr_prefix}{api_key}"

        user_message = str(state.get(input_key, "")) if input_key else str(state)

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        tool_defs: list[dict] = list(tools_cfg)
        if use_tools:
            for t in ctx.tools.values():
                tool_defs.append(self._tool_to_schema(t))

        body: dict = {"model": model, "messages": messages}
        if tool_defs:
            body["tools"] = tool_defs
        if response_format:
            body["response_format"] = response_format
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        has_tools = bool(tool_defs)
        content = ""

        if stream and not has_tools:
            content = await self._post_llm_stream(
                resolved_url, resolved_path, headers, body, on_token
            )
        else:
            for _round in range(self._MAX_TOOL_ROUNDS if has_tools else 1):
                data = await self._post_llm(resolved_url, resolved_path, headers, body)
                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                content = self._extract_content(
                    data, provider_key, response_path, msg.get("content", "")
                )
                tool_calls = msg.get("tool_calls")

                if not has_tools or not tool_calls:
                    break

                body["messages"].append(msg)
                for tc in tool_calls:
                    tc_result = await self._execute_tool_call(tc, ctx)
                    body["messages"].append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tc_result,
                    })

        return {output_key: content or ""}

    async def _post_llm(
        self, base_url: str, path: str, headers: dict, body: dict
    ) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{base_url}{path}", headers=headers, json=body
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
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments for tool '{name}'"

        tool = ctx.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"

        try:
            result = await tool.arun(**args)
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
        sig = inspect.signature(tool.run)
        try:
            hints = typing.get_type_hints(tool.run)
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
