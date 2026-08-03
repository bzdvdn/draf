"""The Harness — transport + agent loop for a single model."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import time
import typing
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass, field

import httpx

from draf.harness.context import trim_messages
from draf.harness.formats import (
    _anthropic_to_message,
    _tool_call_parts,
    extract_content,
    extract_message,
    extract_usage,
    parse_text_tool_call,
)
from draf.harness.providers import (
    _EXPLICIT_LIMITS,
    _PROVIDER_SEMAPHORES,
    PROVIDER_DEFAULTS,
    resolve_provider,
)
from draf.harness.schema import tool_to_schema
from draf.harness.tools import execute_tool_calls
from draf.tool.tool import Tool


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
