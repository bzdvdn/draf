"""Agent harness — reusable model↔tool loop.

A harness owns the transport and provider plumbing for one model and
drives the agent loop: call the model, execute requested tools, feed
the results back into the conversation.  It is shared by the
:class:`~draf.node.llm.LLM` node (internal multi-round loop) and the
:class:`~draf.node.agent.ReActAgent` (one step per graph round, so the
loop stays visible as topology).  Tools are ordinary
:class:`~draf.tool.Tool` instances keyed by name, so MCP tools and
built-in tools work unchanged.

    Behaviour can be parameterised through the constructor / ``from_config``:

    - ``max_rounds`` — stop the ``run()`` loop after this many model calls.
    - ``stop_when(messages)`` — extra termination predicate.
    - ``parse_text_tool_calls`` — decode tool calls embedded in plain text
      (local models often skip the structured ``tool_calls`` field).
    - ``tool_error_mode`` — ``"message"`` (default, errors become tool
      messages) or ``"raise"`` (a tool failure propagates, e.g. into an
      ``__error__`` edge).
    - ``tool_timeout`` — per-tool execution timeout in seconds.
    - ``tool_retries`` — extra attempts per tool call after a failure.
    - ``max_retries`` — retry failed HTTP requests (429/5xx/timeouts).
    - ``retry_on`` — status codes / error types worth retrying.
    - ``fallbacks`` — list of fallback model names used when the primary
      transport fails (provider failover).
    - ``max_total_tokens`` — stop the loop once total prompt+completion
      tokens exceed this budget.
    - ``max_context_tokens`` / ``max_context_messages`` — trim the
      conversation history before each model call to fit these limits.
    - ``cache`` — cache model responses keyed by request so re-runs /
      checkpoint resumes do not pay for the same call twice.
    - ``on_tool_call`` — async hook ``(name, args) -> Awaitable[None]``
      invoked before each tool executes (approval/auditing).
    - ``on_step`` / ``on_llm`` / ``on_token`` — observability hooks.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import time
import typing
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass, field

import httpx

from draf.errors import WorkflowError
from draf.schema import extract_json_object, _py_to_schema
from draf.tool.tool import Tool, coerce_args

PROVIDER_DEFAULTS = {
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
    # OpenAI-compatible endpoints.
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "TOGETHER_API_KEY",
        "chat_path": "/chat/completions",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "GROQ_API_KEY",
        "chat_path": "/chat/completions",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "OPENROUTER_API_KEY",
        "chat_path": "/chat/completions",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "GEMINI_API_KEY",
        "chat_path": "/chat/completions",
    },
    # Custom OpenAI-compatible endpoint (e.g. vLLM, LM Studio, Azure).
    "openai_compatible": {
        "base_url": "",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "api_key_env": "OPENAI_API_KEY",
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

# Global per-provider concurrency guards.  Shared across harness
# instances so parallel branches (each with its own Harness) throttle
# model traffic together instead of blowing past provider rate limits.
_PROVIDER_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
# Providers with an explicit global cap (authoritative over max_parallel).
_EXPLICIT_LIMITS: dict[str, int] = {}


def set_provider_concurrency(provider: str, limit: int) -> None:
    """Globally cap concurrent model calls for *provider*.

    Overrides any per-harness ``max_parallel`` for that provider.
    Pass ``limit <= 0`` to remove the cap.
    """
    provider = provider.lower()
    if limit <= 0:
        _EXPLICIT_LIMITS.pop(provider, None)
        _PROVIDER_SEMAPHORES.pop(provider, None)
    else:
        _EXPLICIT_LIMITS[provider] = limit
        _PROVIDER_SEMAPHORES[provider] = asyncio.Semaphore(limit)


def provider_concurrency(provider: str) -> int | None:
    """Return the current global concurrency limit for *provider* (if any)."""
    sem = _PROVIDER_SEMAPHORES.get(provider.lower())
    return sem._value if sem is not None else None


def _ms(start: float) -> float:
    """Elapsed milliseconds since *start* (a ``time.monotonic()`` stamp)."""
    return (time.monotonic() - start) * 1000.0


def _opt_float(value: typing.Any) -> float | None:
    """Parse *value* as ``float`` or return ``None`` when empty."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: typing.Any) -> int | None:
    """Parse *value* as ``int`` or return ``None`` when empty."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _py_type_to_json(tp: type) -> str:
    return _JSON_TYPE_MAP.get(tp, "string")


def _py_type_to_schema(tp: typing.Any) -> dict:
    """Convert a Python type hint to a (possibly nested) JSON Schema.

    Falls back to a flat ``{"type": ...}`` mapping for types the
    schema builder does not understand.
    """
    try:
        schema = _py_to_schema(tp)
    except Exception:
        schema = {}
    if isinstance(schema, dict) and schema:
        return schema
    return {"type": _py_type_to_json(tp)}


def tool_to_schema(tool: Tool) -> dict:
    """Convert a :class:`~draf.tool.Tool` to an OpenAI-style function schema.

    Uses the tool's ``schema`` attribute when set (e.g. by MCP tools);
    otherwise infers ``parameters`` from the ``run``/``arun`` signature.
    Nested type hints (``list[dict]``, ``dict[str, str]``, dataclasses,
    ``TypedDict``) expand to nested JSON Schemas.
    """
    provider_schema = tool.schema
    if isinstance(provider_schema, dict):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": provider_schema,
            },
        }
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
        prop: dict = _py_type_to_schema(hints.get(pname, str))
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


def extract_message(data: dict) -> dict:
    """Normalise response formats to ``{role, content, tool_calls}``.

    Handles OpenAI (``data["choices"][0]["message"]``) and
    Ollama (``data["message"]`` at root).
    """
    msg = data.get("choices", [{}])[0].get("message", {})
    if not msg and "message" in data:
        msg = data["message"]
    return msg


def _anthropic_to_message(data: dict) -> dict:
    """Convert an Anthropic response into the OpenAI-style message shape.

    Anthropic returns content blocks (``text`` / ``tool_use``); this
    folds them into ``{role, content, tool_calls}`` so the rest of the
    pipeline is provider-agnostic.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in data.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
    msg: dict = {"role": "assistant", "content": "\n".join(text_parts)}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def parse_text_tool_call(content: str) -> tuple[str, dict] | None:
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
        obj = extract_json_object(content, brace)
        if obj is None:
            continue
        try:
            args = json.loads(obj)
        except json.JSONDecodeError:
            args = {}
        return name, args
    return name, {}


def extract_usage(data: dict) -> tuple[int, int]:
    """Extract ``(prompt_tokens, completion_tokens)`` from an LLM response.

    Handles both OpenAI-style (``data["usage"]``) and Ollama-style
    (``data["prompt_eval_count"]`` / ``data["eval_count"]``) formats.
    """
    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None:
        prompt = data.get("prompt_eval_count")
    if completion is None:
        completion = data.get("eval_count")
    if prompt is None:
        prompt = usage.get("input_tokens")
    if completion is None:
        completion = usage.get("output_tokens")
    return int(prompt or 0), int(completion or 0)


def extract_content(
    data: dict, provider: str, path: str = "", fallback: str = ""
) -> str:
    """Extract the assistant text from a response.

    *path* is a dot-separated path into *data*; otherwise provider-aware
    extraction (Anthropic content blocks, Ollama root ``message``).
    """
    if path:
        parts = path.split(".")
        val: typing.Any = data
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


def resolve_provider(
    model: str, provider: str | None = None, default_provider: str | None = None
) -> str:
    """Resolve a provider key from an explicit value or the model name."""
    p = provider or default_provider
    if p:
        return p.lower()
    detected = model.split("-")[0].split("/")[0]
    return detected.lower()


def _tool_call_parts(tc: dict) -> tuple[str, str, str]:
    """Split a tool call into ``(name, raw_args, call_id)``.

    Accepts both the OpenAI shape (``{"function": {...}}``) and the
    graph-signal shape (``{"name", "args", "id"}``).
    """
    if "function" in tc:
        fn = tc["function"] or {}
        raw = fn.get("arguments", "{}")
        if isinstance(raw, dict):
            raw = json.dumps(raw)
        return fn.get("name", ""), raw, tc.get("id", "")
    raw = tc.get("args", "{}")
    if isinstance(raw, dict):
        raw = json.dumps(raw)
    return tc.get("name", ""), raw, tc.get("id", "")


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------


def _content_tokens(content: typing.Any) -> int:
    """Rough token estimate for a message payload (~4 chars per token)."""
    if isinstance(content, str):
        return max(1, len(content) // 4)
    try:
        return max(1, len(json.dumps(content)) // 4)
    except TypeError:
        return 1


def _estimate_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += _content_tokens(content)
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                total += _content_tokens(block.get("text", ""))
    return total


def trim_messages(
    messages: list[dict],
    max_tokens: int | None = None,
    max_messages: int | None = None,
) -> list[dict]:
    """Trim *messages* down to fit context limits.

    The leading ``system`` message (if any) is always preserved; older
    messages are dropped from the front of the conversation until the
    estimated token count and message count fit the limits.

    Args:
        messages: The conversation history.
        max_tokens: Maximum estimated tokens to keep.
        max_messages: Maximum number of messages to keep.

    Returns:
        A new list of messages, trimmed from the front (system kept).
    """
    if not messages:
        return []
    if max_messages is not None and max_messages <= 0:
        return messages
    if max_tokens is not None and max_tokens <= 0:
        return messages

    system: list[dict] = []
    body: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            system.append(msg)
        else:
            body.append(msg)

    if max_messages is not None and len(body) > max_messages:
        body = body[-max_messages:]
    if max_tokens is not None and _estimate_tokens(messages) > max_tokens:
        while body and _estimate_tokens(system + body) > max_tokens:
            body.pop(0)
    return system + body


class ContextLimitError(WorkflowError):
    """Raised when a conversation cannot fit the configured context limits."""


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
            f"\n[draf] approve tool call '{name}' with args {json.dumps(args)}? [y/N] "
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


async def _run_one_tool_call(
    tc: dict,
    tools: Mapping[str, Tool],
    tool_error_mode: str,
    timeout: float | None = None,
    tool_retries: int = 0,
    approver: typing.Any = None,
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
    for attempt in range(attempts):
        try:
            coro = tool.arun(**coerce_args(tool, args))
            if timeout and timeout > 0:
                coro = asyncio.wait_for(coro, timeout=timeout)
            result = await coro
            return str(result) if result is not None else ""
        except asyncio.TimeoutError:
            return f"Error calling '{name}': timed out after {timeout}s"
        except Exception as exc:
            if tool_error_mode == "raise" or attempt == attempts - 1:
                if tool_error_mode == "raise":
                    raise
                return f"Error calling '{name}': {exc}"
    return f"Error calling '{name}': failed"


async def execute_tool_calls(
    tool_calls: list[dict],
    tools: Mapping[str, Tool],
    tool_error_mode: str = "message",
    timeout: float | None = None,
    tool_retries: int = 0,
    approver: typing.Any = None,
) -> list[str]:
    """Execute *tool_calls* against *tools* in parallel.

    Each call resolves to a result string (errors become ``"Error ..."``
    messages unless *tool_error_mode* is ``"raise"``).  Each call is
    retried up to *tool_retries* times on failure and bounded by *timeout*
    seconds when set.  An optional *approver* gates each call before it
    runs (see :func:`resolve_approval`); non-``"approve"`` decisions
    short-circuit the call with a "not approved" message.

    Args:
        tool_calls: List of tool-call dicts.
        tools: Tool registry (name -> ``Tool``).
        tool_error_mode: ``"message"`` or ``"raise"``.
        timeout: Per-tool timeout in seconds (``None`` = no limit).
        tool_retries: Extra attempts per tool call after a failure.
        approver: Approval policy (string or callable).
    """
    if not tool_calls:
        return []
    return await asyncio.gather(
        *(
            _run_one_tool_call(
                tc, tools, tool_error_mode, timeout, tool_retries, approver
            )
            for tc in tool_calls
        )
    )


@dataclass
class ModelReply:
    """A single model call's result."""

    data: dict
    message: dict
    content: str
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False


@dataclass
class Step:
    """One iteration of the agent loop (model call + any tool execution)."""

    messages: list[dict]
    content: str
    tool_calls: list[dict]
    reply: ModelReply

    @property
    def wants_tool(self) -> bool:
        """Whether the step ended requesting more tool execution."""
        return bool(self.tool_calls)


class Harness:
    """Transport + loop for one model, reusable across nodes and flows.

    Args:
        model: Model name (e.g. ``gpt-4``, ``llama3.1:8b``).
        provider: Provider name; falls back to *default_provider*, then
            auto-detection from *model*.
        base_url / api_key_env / chat_path / auth_header / auth_prefix:
            Overrides for the provider defaults.
        timeout: HTTP timeout in seconds.
        max_rounds: Maximum model calls for :meth:`run`.
        parse_text_tool_calls: Decode text-embedded tool calls.
        tool_error_mode: ``"message"`` or ``"raise"``.
        stop_when: Optional ``(messages) -> bool`` termination predicate.
        on_step: Async callback ``(Step) -> None`` after each iteration.
        on_llm: Async callback ``(provider, model, prompt_tokens,
            completion_tokens, latency_ms) -> None`` after each model call.
        on_token: Token callback for streaming.
        temperature / max_tokens / response_format: Default body extras.
        stream: Stream tokens by default (disabled while tools are active).
        default_provider: Fallback provider (e.g. ``set_defaults``).
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        base_url: str = "",
        api_key_env: str = "",
        chat_path: str = "",
        auth_header: str = "",
        auth_prefix: str = "",
        timeout: float = 120,
        max_rounds: int = 10,
        parse_text_tool_calls: bool = True,
        tool_error_mode: str = "message",
        tool_timeout: float | None = None,
        tool_retries: int = 0,
        max_retries: int = 2,
        retry_on: tuple[int, ...] = (429, 500, 502, 503, 504),
        fallbacks: list[str] | None = None,
        tool_approval: typing.Any = None,
        max_total_tokens: int | None = None,
        max_context_tokens: int | None = None,
        max_context_messages: int | None = None,
        max_parallel: int | None = None,
        stop_when: Callable[[list[dict]], bool] | None = None,
        on_step: Callable[[Step], Awaitable[None]] | None = None,
        on_llm: Callable[[str, str, int, int, float], Awaitable[None]] | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[str, dict], Awaitable[None]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        stream: bool = False,
        cache: "MutableMapping[str, str] | bool | None" = None,
        default_provider: str | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self.max_rounds = max_rounds
        self.parse_text_tool_calls = parse_text_tool_calls
        self.tool_error_mode = tool_error_mode
        self.tool_timeout = tool_timeout
        self.tool_retries = tool_retries
        self.max_retries = max_retries
        self.retry_on = tuple(retry_on or ())
        self.fallbacks = list(fallbacks or [])
        self.max_parallel: int | None = max_parallel
        self._tool_approval = tool_approval
        self.max_total_tokens = max_total_tokens
        self.max_context_tokens = max_context_tokens
        self.max_context_messages = max_context_messages
        self.stop_when = stop_when
        self.on_step = on_step
        self.on_llm = on_llm
        self.on_token = on_token
        self.on_tool_call = on_tool_call
        self.stream = stream
        self._cache: MutableMapping[str, str] | None = None
        if isinstance(cache, bool):
            if cache:
                self._cache = {}
        elif cache is not None:
            self._cache = cache

        self.provider_key = resolve_provider(model, provider, default_provider)
        defaults = PROVIDER_DEFAULTS.get(self.provider_key, PROVIDER_DEFAULTS["openai"])

        resolved_url = base_url or os.environ.get(
            f"{self.provider_key.upper()}_BASE_URL", defaults["base_url"]
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
            hdr_prefix = auth_prefix or defaults["auth_prefix"]
            headers[hdr_name] = f"{hdr_prefix}{api_key}"

        self._url = f"{resolved_url}{resolved_path}"
        self._headers = headers

        self._body_extra: dict = {}
        if temperature is not None:
            self._body_extra["temperature"] = temperature
        if max_tokens is not None:
            self._body_extra["max_tokens"] = max_tokens
        if response_format is not None:
            self._body_extra["response_format"] = response_format

        # Provider failover: build fallback transports lazily.  Each
        # fallback is described by (model, url, headers) and tried in
        # order when the primary request fails after all retries.
        self._fallback_transports: list[tuple[str, str, dict]] = []
        for fb_model in self.fallbacks:
            fb_provider = resolve_provider(fb_model, default_provider=default_provider)
            fb_defaults = PROVIDER_DEFAULTS.get(
                fb_provider, PROVIDER_DEFAULTS["openai"]
            )
            fb_url = base_url or os.environ.get(
                f"{fb_provider.upper()}_BASE_URL", fb_defaults["base_url"]
            )
            fb_env = api_key_env or fb_defaults["api_key_env"]
            fb_path = chat_path or fb_defaults["chat_path"]
            fb_key = ""
            if fb_env:
                fb_key = os.environ.get(fb_env, "")
            if not fb_key:
                fb_key = os.environ.get("LLM_API_KEY", "")
            fb_headers = {"Content-Type": "application/json"}
            fb_hdr = auth_header or fb_defaults["auth_header"]
            if fb_hdr and fb_key:
                fb_prefix = auth_prefix or fb_defaults["auth_prefix"]
                fb_headers[fb_hdr] = f"{fb_prefix}{fb_key}"
            self._fallback_transports.append(
                (fb_model, f"{fb_url}{fb_path}", fb_headers)
            )

        # Token budget tracking across calls.
        self.total_tokens = 0

        # Register the per-provider concurrency guard (grows global cap).
        self._concurrency_semaphore()

    @classmethod
    def from_config(
        cls, cfg: dict, *, default_provider: str | None = None
    ) -> "Harness":
        """Build a harness from a node config dict.

        Recognises the transport keys shared by ``LLM`` and
        ``ReActAgent`` plus the loop knobs ``max_tool_rounds``,
        ``tool_error_mode``, ``parse_text_tool_calls``, ``tool_timeout``,
        ``tool_retries``, ``max_retries``, ``fallbacks``,
        ``max_total_tokens``, ``max_context_tokens`` and
        ``max_context_messages``.
        """
        return cls(
            model=str(cfg.get("model", "gpt-4")),
            provider=cfg.get("provider"),
            base_url=cfg.get("base_url") or "",
            api_key_env=cfg.get("api_key_env") or "",
            chat_path=cfg.get("chat_path") or "",
            auth_header=cfg.get("auth_header") or "",
            auth_prefix=cfg.get("auth_prefix") or "",
            timeout=float(cfg.get("timeout") or 120),
            max_rounds=int(cfg.get("max_tool_rounds") or 10),
            parse_text_tool_calls=bool(cfg.get("parse_text_tool_calls", True)),
            tool_error_mode=str(cfg.get("tool_error_mode", "message")),
            tool_timeout=_opt_float(cfg.get("tool_timeout")),
            tool_retries=int(cfg.get("tool_retries") or 0),
            max_retries=int(cfg.get("max_retries") or 2),
            retry_on=tuple(int(x) for x in cfg.get("retry_on") or ())
            or (429, 500, 502, 503, 504),
            fallbacks=cfg.get("fallbacks"),
            tool_approval=cfg.get("tool_approval"),
            max_total_tokens=_opt_int(cfg.get("max_total_tokens")),
            max_context_tokens=_opt_int(cfg.get("max_context_tokens")),
            max_context_messages=_opt_int(cfg.get("max_context_messages")),
            max_parallel=_opt_int(cfg.get("max_parallel")),
            stop_when=cfg.get("stop_when"),
            on_step=cfg.get("on_step"),
            on_llm=cfg.get("on_llm"),
            on_token=cfg.get("on_token"),
            on_tool_call=cfg.get("on_tool_call"),
            temperature=cfg.get("temperature"),
            max_tokens=cfg.get("max_tokens"),
            response_format=cfg.get("response_format"),
            stream=bool(cfg.get("stream", False)),
            cache=cfg.get("cache"),
            default_provider=default_provider,
        )

    def _body(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        if self.provider_key == "anthropic":
            return self._anthropic_body(messages, tools)
        body: dict = {"model": self.model, "messages": messages, **self._body_extra}
        if tools:
            body["tools"] = tools
        return body

    def _anthropic_body(self, messages: list[dict], tools: list[dict] | None) -> dict:
        """Build an Anthropic ``/messages`` request body.

        Splits ``system`` out to the top level, converts tool results and
        assistant ``tool_calls`` into content blocks, and rewrites the
        tool schemas into Anthropic's ``input_schema`` shape.
        """
        system = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        body: dict = {
            "model": self.model,
            "messages": [
                self._to_anthropic_message(m)
                for m in messages
                if m.get("role") != "system"
            ],
            "max_tokens": self._body_extra.get("max_tokens") or 1024,
        }
        if self._body_extra.get("temperature") is not None:
            body["temperature"] = self._body_extra["temperature"]
        if self._body_extra.get("response_format") is not None:
            body["response_format"] = self._body_extra["response_format"]
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [self._to_anthropic_tool(t) for t in tools]
        return body

    @staticmethod
    def _to_anthropic_message(msg: dict) -> dict:
        """Convert an OpenAI-shaped message into an Anthropic one."""
        role = msg.get("role")
        if role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }
                ],
            }
        if role == "assistant" and msg.get("tool_calls"):
            blocks: list[dict] = []
            content = msg.get("content")
            if content:
                blocks.append({"type": "text", "text": str(content)})
            for tc in msg["tool_calls"]:
                name, raw, call_id = _tool_call_parts(tc)
                try:
                    args = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    args = {}
                blocks.append(
                    {"type": "tool_use", "id": call_id, "name": name, "input": args}
                )
            return {"role": "assistant", "content": blocks}
        return {"role": role, "content": msg.get("content", "")}

    @staticmethod
    def _to_anthropic_tool(tool: dict) -> dict:
        """Convert an OpenAI function schema into an Anthropic tool schema."""
        fn = tool.get("function", tool)
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        }

    def _is_retryable(self, exc: Exception) -> bool:
        """Whether an HTTP exception should be retried per *retry_on*."""
        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True
        status = getattr(exc, "response", None)
        code = getattr(status, "status_code", None)
        return code is not None and code in self.retry_on

    def _concurrency_semaphore(self) -> asyncio.Semaphore | None:
        """Global semaphore for this provider.

        An explicit cap (``set_provider_concurrency``) is authoritative;
        otherwise the shared semaphore grows to the largest ``max_parallel``
        any harness has configured for the provider.
        """
        key = self.provider_key
        if key in _EXPLICIT_LIMITS:
            return _PROVIDER_SEMAPHORES.get(key)
        if self.max_parallel and self.max_parallel > 0:
            sem = _PROVIDER_SEMAPHORES.get(key)
            if sem is None or sem._value < self.max_parallel:
                sem = asyncio.Semaphore(self.max_parallel)
                _PROVIDER_SEMAPHORES[key] = sem
            return sem
        return None

    async def _post_with_retries(
        self, url: str, headers: dict, body: dict, *, allow_fallback: bool = True
    ) -> dict:
        """POST *body* with retries + backoff, then fail over on repeated errors."""
        sem = self._concurrency_semaphore()
        if sem is not None:
            async with sem:
                return await self._post_with_retries_impl(
                    url, headers, body, allow_fallback=allow_fallback
                )
        return await self._post_with_retries_impl(
            url, headers, body, allow_fallback=allow_fallback
        )

    async def _post_with_retries_impl(
        self, url: str, headers: dict, body: dict, *, allow_fallback: bool = True
    ) -> dict:
        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=body)
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:  # noqa: BLE001 — retry policy drives handling
                    last_exc = exc
                    if not self._is_retryable(exc) or attempt >= self.max_retries:
                        break
                    await asyncio.sleep(min(4.0, 0.5 * (2**attempt)))
        # Primary transport exhausted — try fallback models (once).
        if allow_fallback and last_exc is not None and self._fallback_transports:
            for fb_model, fb_url, fb_headers in self._fallback_transports:
                try:
                    return await self._post_with_retries_impl(
                        fb_url,
                        fb_headers,
                        {**body, "model": fb_model},
                        allow_fallback=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def _post_stream_with_retries(
        self, url: str, headers: dict, body: dict, *, allow_fallback: bool = True
    ) -> str:
        """Stream a POST response with retries + backoff + failover."""
        sem = self._concurrency_semaphore()
        if sem is not None:
            async with sem:
                return await self._post_stream_with_retries_impl(
                    url, headers, body, allow_fallback=allow_fallback
                )
        return await self._post_stream_with_retries_impl(
            url, headers, body, allow_fallback=allow_fallback
        )

    async def _post_stream_with_retries_impl(
        self, url: str, headers: dict, body: dict, *, allow_fallback: bool = True
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._stream_once(url, headers, body)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not self._is_retryable(exc) or attempt >= self.max_retries:
                    break
                await asyncio.sleep(min(4.0, 0.5 * (2**attempt)))
        if allow_fallback and last_exc is not None and self._fallback_transports:
            for fb_model, fb_url, fb_headers in self._fallback_transports:
                try:
                    return await self._stream_once(
                        fb_url, fb_headers, {**body, "model": fb_model}
                    )
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def _stream_once(self, url: str, headers: dict, body: dict) -> str:
        content = ""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                    elif line.startswith("{"):
                        data = line.strip()
                    else:
                        continue
                    if data == "[DONE]":
                        break
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    token = self._stream_token(chunk)
                    if token:
                        content += token
                        if self.on_token:
                            result = self.on_token(token)
                            if inspect.isawaitable(result):
                                await result
        return content

    def _stream_token(self, chunk: dict) -> str:
        """Extract a text delta from a streaming chunk (provider-aware)."""
        if self.provider_key == "anthropic":
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta":
                return str(delta.get("text", ""))
            return ""
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        token = delta.get("content", "")
        if not token:
            token = (chunk.get("message") or {}).get("content", "")
        return str(token)

    async def _post(self, body: dict) -> dict:
        return await self._post_with_retries(
            self._url, self._headers, {**body, "stream": False}
        )

    async def _post_stream(self, body: dict) -> str:
        body = {**body, "stream": True}
        return await self._post_stream_with_retries(self._url, self._headers, body)

    def _cache_key(self, body: dict) -> str:
        """Hash of the request that identifies a cacheable model call."""
        payload = json.dumps(body, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return f"{self.provider_key}:{self.model}:{digest}"

    async def call(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        stream: bool | None = None,
        content_path: str = "",
    ) -> ModelReply:
        """One model call.

        Args:
            messages: Message history.
            tools: Tool schemas to attach (disables streaming).
            stream: Force streaming on/off (defaults to *self.stream* and
                automatically off when *tools* are attached).
            content_path: Dot-separated path for content extraction.

        Returns:
            A :class:`ModelReply` (``cached=True`` when served from cache).
        """
        body = self._body(messages, tools)
        use_stream = stream if stream is not None else (self.stream and not tools)
        t0 = time.monotonic()

        cached = False
        cache_key: str | None = None
        if not use_stream and self._cache is not None:
            cache_key = self._cache_key(body)
            hit = self._cache.get(cache_key)
            if hit is not None:
                data = json.loads(hit) if isinstance(hit, str) else hit
                cached = True

        if use_stream:
            content = await self._post_stream(body)
            msg: dict = {"role": "assistant", "content": content}
            data = {"message": msg}
            usage: dict = {"prompt": 0, "completion": 0}
        else:
            if not cached:
                data = await self._post(body)
                if self._cache is not None and cache_key is not None:
                    self._cache[cache_key] = json.dumps(data, default=str)
            if self.provider_key == "anthropic":
                msg = _anthropic_to_message(data)
                content = msg.get("content", "")
            else:
                msg = extract_message(data)
                content = extract_content(
                    data, self.provider_key, content_path, msg.get("content", "")
                )
            prompt, completion = extract_usage(data)
            usage = {"prompt": prompt, "completion": completion}
            if self.on_llm:
                await self.on_llm(
                    self.provider_key, self.model, prompt, completion, _ms(t0)
                )
        self.total_tokens += int(usage.get("prompt", 0)) + int(
            usage.get("completion", 0)
        )
        return ModelReply(
            data=data,
            message=msg,
            content=content,
            usage=usage,
            latency_ms=_ms(t0),
            cached=cached,
        )

    async def step(
        self, messages: list[dict], tools: Mapping[str, Tool] | None
    ) -> Step:
        """One iteration: call the model, execute requested tools, feed back.

        Returns a :class:`Step` whose ``messages`` is the updated history
        (assistant message plus any ``tool`` responses).  History is
        trimmed to *max_context_tokens* / *max_context_messages* before
        the call.
        """
        messages = self.manage_context(messages)
        tool_defs = [tool_to_schema(t) for t in tools.values()] if tools else []
        reply = await self.call(messages, tools=tool_defs or None)
        tool_calls = reply.message.get("tool_calls")

        if tool_defs and not tool_calls and self.parse_text_tool_calls:
            parsed = parse_text_tool_call(reply.content)
            if parsed:
                name, args = parsed
                call_id = f"call_{len(messages)}"
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
                reply.message = {**reply.message, "tool_calls": tool_calls}

        new_messages = list(messages)
        if tool_calls:
            if self.on_tool_call is not None:
                for tc in tool_calls:
                    name, raw, _ = _tool_call_parts(tc)
                    try:
                        args = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        args = {}
                    result = self.on_tool_call(name, args)
                    if inspect.isawaitable(result):
                        await result
            new_messages.append(reply.message)
            results = await execute_tool_calls(
                tool_calls,
                tools or {},
                self.tool_error_mode,
                self.tool_timeout,
                self.tool_retries,
                self._tool_approval,
            )
            for tc, res in zip(tool_calls, results):
                new_messages.append(
                    {"role": "tool", "tool_call_id": tc.get("id", ""), "content": res}
                )
            step = Step(
                messages=new_messages,
                content=reply.content,
                tool_calls=tool_calls,
                reply=reply,
            )
        else:
            step = Step(
                messages=new_messages,
                content=reply.content,
                tool_calls=[],
                reply=reply,
            )
        if self.on_step:
            await self.on_step(step)
        return step

    def manage_context(self, messages: list[dict]) -> list[dict]:
        """Trim *messages* to the configured context limits.

        Applies ``max_context_tokens`` / ``max_context_messages``
        (whichever is set).  The leading ``system`` message is preserved.
        """
        if self.max_context_tokens is None and self.max_context_messages is None:
            return messages
        return trim_messages(
            messages,
            max_tokens=self.max_context_tokens,
            max_messages=self.max_context_messages,
        )

    async def run(self, messages: list[dict], tools: Mapping[str, Tool] | None) -> Step:
        """Loop :meth:`step` until a final answer, *stop_when*, or *max_rounds*.

        Stops early when the cumulative token budget (*max_total_tokens*)
        is exceeded.

        Returns the final :class:`Step` (its ``content`` holds the answer;
        ``messages`` holds the full history).
        """
        step = await self.step(messages, tools)
        for _ in range(1, self.max_rounds):
            if not step.wants_tool:
                break
            if self.stop_when is not None and self.stop_when(step.messages):
                break
            if (
                self.max_total_tokens is not None
                and self.total_tokens >= self.max_total_tokens
            ):
                break
            step = await self.step(step.messages, tools)
        return step
