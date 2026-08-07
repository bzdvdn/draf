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

from teff.errors import redact
from teff.harness.context import trim_messages
from teff.harness.formats import (
    _anthropic_to_message,
    _tool_call_parts,
    extract_content,
    extract_message,
    extract_usage,
    normalize_text_tool_calls,
)
from teff.harness.schema import tool_to_schema
from teff.harness.tools import execute_tool_calls
from teff.logging import get_logger
from teff.provider import (
    Provider,
    ProviderRegistry,
    resolve_provider,
    resolve_provider_entry,
)
from teff.provider.providers import (
    _EXPLICIT_LIMITS,
    _PROVIDER_ACTIVE,
    _PROVIDER_LIMITS,
    _PROVIDER_SEMAPHORES,
)
from teff.tool.tool import Tool

log = get_logger(__name__)

#: Shared request cache used when ``cache=True`` so re-runs / checkpoint
#: resumes across distinct :class:`Harness` instances hit the same store.
_DEFAULT_CACHE: dict[str, str] = {}


def _last_user_message(messages: list[dict]) -> str:
    """Return the most recent ``user`` message content (empty if none)."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _truncate(value: str, cap: int | None = None) -> str:
    """Truncate *value* to ``TEFF_LOG_LLM_CHARS`` chars (default 2000)."""
    if value is None:
        return ""
    if cap is None:
        try:
            cap = int(os.environ.get("TEFF_LOG_LLM_CHARS", "2000"))
        except ValueError:
            cap = 2000
    if cap <= 0 or len(value) <= cap:
        return value
    return f"{value[:cap]}...(truncated {len(value) - cap} chars)"


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


def _cfg_int(cfg: dict, key: str, default: int) -> int:
    """Parse *cfg[key]* as ``int``, using *default* when absent or ``None``.

    Unlike the ``or``-based coercion, an explicit ``0`` is honoured rather
    than silently replaced by *default* (e.g. ``max_retries: 0`` disables
    retries).
    """
    value = cfg.get(key)
    if value is None:
        return default
    return int(value)


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
        provider: Provider name (``"openai"``, ``"ollama"``, etc.).
            Falls back to *default_provider* when unset.
        base_url / api_key_env / chat_path / auth_header / auth_prefix:
            Overrides for the provider defaults.
        providers: Optional ``{name: Provider}`` map or
            :class:`~teff.provider.ProviderRegistry` (custom providers
            declared in a workflow / passed to ``graph.run``).  Entries
            are resolved before the built-in presets.
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
        default_provider: Fallback provider name (the graph-level default,
            e.g. ``Graph(default_provider="ollama")`` or a workflow
            ``default_provider:``).
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        base_url: str = "",
        api_key_env: str = "",
        chat_path: str = "",
        auth_header: str = "",
        auth_prefix: str = "",
        timeout: float | None = 120,
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
        on_llm_payload: Callable[
            [str, str, list[dict], str, dict, float, bool], Awaitable[None]
        ]
        | None = None,
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
        self.timeout = timeout or 120
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
        self.on_llm_payload = on_llm_payload
        self.on_token = on_token
        self.on_tool_call = on_tool_call
        self.stream = stream
        self._cache: MutableMapping[str, str] | None = None
        if isinstance(cache, bool):
            if cache:
                # ``True`` shares one process-wide store so distinct harnesses
                # (per-node/per-run instances) reuse the same responses.
                self._cache = _DEFAULT_CACHE
        elif cache is not None:
            self._cache = cache

        self.provider_key = resolve_provider(provider, default_provider)
        entry = resolve_provider_entry(self.provider_key, providers)
        self.type = entry.type
        if not self.timeout or self.timeout <= 0:
            self.timeout = entry.timeout or 120

        resolved_url = base_url or os.environ.get(
            f"{self.provider_key.upper()}_BASE_URL", entry.base_url
        )
        resolved_env = api_key_env or entry.api_key_env
        resolved_path = chat_path or entry.chat_path

        api_key = ""
        if resolved_env:
            api_key = os.environ.get(resolved_env, "")
        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", "")

        headers = {"Content-Type": "application/json"}
        hdr_name = auth_header or entry.auth_header
        if hdr_name and api_key:
            hdr_prefix = auth_prefix or entry.auth_prefix
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
            fb_provider = self.provider_key
            fb_entry = resolve_provider_entry(fb_provider, providers)
            fb_url = base_url or os.environ.get(
                f"{fb_provider.upper()}_BASE_URL", fb_entry.base_url
            )
            fb_env = api_key_env or fb_entry.api_key_env
            fb_path = chat_path or fb_entry.chat_path
            fb_key = ""
            if fb_env:
                fb_key = os.environ.get(fb_env, "")
            if not fb_key:
                fb_key = os.environ.get("LLM_API_KEY", "")
            fb_headers = {"Content-Type": "application/json"}
            fb_hdr = auth_header or fb_entry.auth_header
            if fb_hdr and fb_key:
                fb_prefix = auth_prefix or fb_entry.auth_prefix
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
        cls,
        cfg: dict,
        *,
        default_provider: str | None = None,
        default_model: str | None = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
    ) -> "Harness":
        """Build a harness from a node config dict.

        Recognises the transport keys shared by ``LLM`` and
        ``ReActAgent`` plus the loop knobs ``max_tool_rounds``,
        ``tool_error_mode``, ``parse_text_tool_calls``, ``tool_timeout``,
        ``tool_retries``, ``max_retries``, ``fallbacks``,
        ``max_total_tokens``, ``max_context_tokens`` and
        ``max_context_messages``.

        *providers* is an optional ``{name: Provider}`` map or
        :class:`~teff.provider.ProviderRegistry` (custom providers from
        the workflow) consulted before the built-in presets.

        The model name comes from ``cfg["model"]`` or, when absent,
        *default_model* (the graph-level default).  When neither is set a
        :class:`ConfigError` is raised — there is no silent model default.
        """
        model = cfg.get("model") or default_model
        if not model:
            from teff.errors import ConfigError

            raise ConfigError(
                "no model configured: set `model=` on the node or pass "
                "`default_model=` to the graph / `default_model:` in the "
                "workflow"
            )
        return cls(
            model=str(model),
            provider=cfg.get("provider"),
            providers=providers,
            base_url=cfg.get("base_url") or "",
            api_key_env=cfg.get("api_key_env") or "",
            chat_path=cfg.get("chat_path") or "",
            auth_header=cfg.get("auth_header") or "",
            auth_prefix=cfg.get("auth_prefix") or "",
            timeout=_opt_float(cfg.get("timeout")),
            max_rounds=_cfg_int(cfg, "max_tool_rounds", 10),
            parse_text_tool_calls=bool(cfg.get("parse_text_tool_calls", True)),
            tool_error_mode=str(cfg.get("tool_error_mode", "message")),
            tool_timeout=_opt_float(cfg.get("tool_timeout")),
            tool_retries=_cfg_int(cfg, "tool_retries", 0),
            max_retries=_cfg_int(cfg, "max_retries", 2),
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
        if self.type == "anthropic_compatible":
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
        any harness has configured for the provider.  Growth only replaces an
        idle semaphore, so in-flight requests never exceed the new cap
        (replacing a contended semaphore would let old + new holders run
        concurrently past the limit).
        """
        key = self.provider_key
        if key in _EXPLICIT_LIMITS:
            return _PROVIDER_SEMAPHORES.get(key)
        if self.max_parallel and self.max_parallel > 0:
            sem = _PROVIDER_SEMAPHORES.get(key)
            current = _PROVIDER_LIMITS.get(key, 0)
            if current >= self.max_parallel:
                return sem
            # Grow only while no request is in flight (idle).  Replacing a
            # contended semaphore would let old + new holders run past the cap.
            if sem is None or _PROVIDER_ACTIVE.get(key, 0) == 0:
                sem = asyncio.Semaphore(self.max_parallel)
                _PROVIDER_SEMAPHORES[key] = sem
                _PROVIDER_LIMITS[key] = self.max_parallel
                _PROVIDER_ACTIVE[key] = 0
            return _PROVIDER_SEMAPHORES[key]
        return None

    async def _post_with_retries(
        self, url: str, headers: dict, body: dict, *, allow_fallback: bool = True
    ) -> tuple[dict, bool]:
        """POST *body* with retries + backoff + failover.

        Returns ``(data, used_fallback)`` where *used_fallback* is ``True``
        when the response came from a fallback model rather than the primary.

        The provider semaphore is held for the whole retry cycle (including
        the backoff sleeps) — a slow/failing request keeps one slot so healthy
        in-flight calls never exceed the agreed cap.
        """
        key = self.provider_key
        sem = self._concurrency_semaphore()
        if sem is None:
            return await self._post_with_retries_impl(
                url, headers, body, allow_fallback=allow_fallback
            )
        _PROVIDER_ACTIVE[key] = _PROVIDER_ACTIVE.get(key, 0) + 1
        try:
            async with sem:
                return await self._post_with_retries_impl(
                    url, headers, body, allow_fallback=allow_fallback
                )
        finally:
            _PROVIDER_ACTIVE[key] = max(0, _PROVIDER_ACTIVE.get(key, 0) - 1)

    async def _post_with_retries_impl(
        self, url: str, headers: dict, body: dict, *, allow_fallback: bool = True
    ) -> tuple[dict, bool]:
        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=body)
                    response.raise_for_status()
                    return response.json(), False
                except Exception as exc:  # noqa: BLE001 — retry policy drives handling
                    last_exc = exc
                    if not self._is_retryable(exc) or attempt >= self.max_retries:
                        break
                    await asyncio.sleep(min(4.0, 0.5 * (2**attempt)))
        # Primary transport exhausted — try fallback models (once each).
        if allow_fallback and last_exc is not None and self._fallback_transports:
            for fb_model, fb_url, fb_headers in self._fallback_transports:
                try:
                    data, _ = await self._post_with_retries_impl(
                        fb_url,
                        fb_headers,
                        {**body, "model": fb_model},
                        allow_fallback=False,
                    )
                    return data, True
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def _post_stream_with_retries(
        self, url: str, headers: dict, body: dict, *, allow_fallback: bool = True
    ) -> tuple[str, dict]:
        """Stream a POST response with retries + backoff + failover.

        Returns ``(content, usage)``; *usage* carries the provider-reported
        token counts when the final chunk included them, otherwise ``{}``.
        """
        key = self.provider_key
        sem = self._concurrency_semaphore()
        if sem is None:
            return await self._post_stream_with_retries_impl(
                url, headers, body, allow_fallback=allow_fallback
            )
        _PROVIDER_ACTIVE[key] = _PROVIDER_ACTIVE.get(key, 0) + 1
        try:
            async with sem:
                return await self._post_stream_with_retries_impl(
                    url, headers, body, allow_fallback=allow_fallback
                )
        finally:
            _PROVIDER_ACTIVE[key] = max(0, _PROVIDER_ACTIVE.get(key, 0) - 1)

    async def _post_stream_with_retries_impl(
        self, url: str, headers: dict, body: dict, *, allow_fallback: bool = True
    ) -> tuple[str, dict]:
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
                    return await self._post_stream_with_retries_impl(
                        fb_url,
                        fb_headers,
                        {**body, "model": fb_model},
                        allow_fallback=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def _stream_once(
        self, url: str, headers: dict, body: dict
    ) -> tuple[str, dict]:
        content = ""
        usage: dict = {}
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
                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict) and chunk_usage:
                        usage = chunk_usage
                    token = self._stream_token(chunk)
                    if token:
                        content += token
                        if self.on_token:
                            result = self.on_token(token)
                            if inspect.isawaitable(result):
                                await result
        return content, usage

    def _stream_token(self, chunk: dict) -> str:
        """Extract a text delta from a streaming chunk (provider-aware)."""
        if self.type == "anthropic_compatible":
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta":
                return str(delta.get("text", ""))
            return ""
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        token = delta.get("content", "")
        if not token:
            token = (chunk.get("message") or {}).get("content", "")
        return str(token)

    async def _post(self, body: dict) -> tuple[dict, bool]:
        return await self._post_with_retries(
            self._url, self._headers, {**body, "stream": False}
        )

    async def _post_stream(self, body: dict) -> tuple[str, dict]:
        body = {**body, "stream": True}
        return await self._post_stream_with_retries(self._url, self._headers, body)

    def _cache_key(self, body: dict) -> str:
        """Hash of the request that identifies a cacheable model call."""
        payload = json.dumps(body, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return f"{self.provider_key}:{self.model}:{digest}"

    @staticmethod
    def _estimate_message_tokens(messages: list[dict]) -> int:
        """Rough prompt-token estimate (~4 chars per token)."""
        total = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    total += max(1, len(str(block.get("text", ""))) // 4)
            else:
                total += max(1, len(str(content)) // 4)
        return total

    def _stream_tokens(
        self, messages: list[dict], content: str, stream_usage: dict
    ) -> tuple[int, int]:
        """Token counts for a streamed call.

        Uses provider-reported usage from the streamed chunks when present;
        otherwise falls back to rough estimates so budgets/hooks still work.
        """
        prompt = stream_usage.get("prompt_tokens") or stream_usage.get("input_tokens")
        completion = stream_usage.get("completion_tokens") or stream_usage.get(
            "output_tokens"
        )
        if prompt is None:
            prompt = self._estimate_message_tokens(messages)
        if completion is None:
            completion = len(content) // 4
        return int(prompt or 0), int(completion or 0)

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
        data: dict = {}
        if not use_stream and self._cache is not None:
            cache_key = self._cache_key(body)
            hit = self._cache.get(cache_key)
            if hit is not None:
                data = json.loads(hit) if isinstance(hit, str) else hit
                cached = True

        if use_stream:
            content, stream_usage = await self._post_stream(body)
            msg: dict = {"role": "assistant", "content": content}
            data = {"message": msg}
            prompt, completion = self._stream_tokens(messages, content, stream_usage)
            usage = {"prompt": prompt, "completion": completion}
            if self.on_llm:
                await self.on_llm(
                    self.provider_key, self.model, prompt, completion, _ms(t0)
                )
        else:
            if not cached:
                data, used_fallback = await self._post(body)
                if (
                    self._cache is not None
                    and cache_key is not None
                    and not used_fallback
                ):
                    # Never cache a fallback model's reply under the primary's key,
                    # otherwise a recovered primary keeps serving stale fallback output.
                    self._cache[cache_key] = json.dumps(data, default=str)
            if self.type == "anthropic_compatible":
                msg = _anthropic_to_message(data)
                content = msg.get("content", "")
            else:
                msg = extract_message(data)
                content = extract_content(
                    data, self.type, content_path, msg.get("content", "")
                )
            prompt, completion = extract_usage(data)
            usage = {"prompt": prompt, "completion": completion}
            if self.on_llm:
                await self.on_llm(
                    self.provider_key, self.model, prompt, completion, _ms(t0)
                )
        log.info(
            "llm_call model=%s provider=%s prompt_tokens=%s completion_tokens=%s latency_ms=%s",
            self.model,
            self.provider_key,
            usage.get("prompt"),
            usage.get("completion"),
            f"{_ms(t0):.0f}",
        )
        log.debug("llm_request %s", _truncate(redact(_last_user_message(messages))))
        log.debug("llm_response %s", _truncate(redact(content)))
        if self.on_llm_payload is not None:
            await self.on_llm_payload(
                self.provider_key,
                self.model,
                messages,
                content,
                usage,
                _ms(t0),
                cached,
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
            tool_calls, reply.message = normalize_text_tool_calls(
                reply.content, reply.message, seq=len(messages)
            )

        new_messages = list(messages)
        if tool_calls:
            log.info("tool_call count=%s", len(tool_calls))
            if self.on_tool_call is not None:
                for tc in tool_calls:
                    name, raw, _ = _tool_call_parts(tc)
                    try:
                        args = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        args = {}
                    log.info(
                        "tool_call tool=%s args=%s",
                        name,
                        _truncate(json.dumps(redact(args), default=str)),
                    )
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
