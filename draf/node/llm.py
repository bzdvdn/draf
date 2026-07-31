"""LLM chat node — multi-provider, tool calling, structured output."""

import json
import typing

from draf.harness import (
    Harness,
    execute_tool_calls,
    extract_usage as _extract_usage,  # noqa: F401  (re-export for tests)
    parse_text_tool_call as _parse_text_tool_call,
    tool_to_schema,
)
from draf.node.node import Node
from draf.prompt import render_template
from draf.tool.tool import Tool
from draf.stream import StreamEvent
from draf.schema import (
    json_schema_from_type,
    parse_json_object,
    validate_json,
)


class StructuredOutputError(ValueError):
    """Raised when an LLM response fails structured-output parsing/validation.

    Attributes:
        schema: The JSON Schema the output was validated against (or ``None``).
        content: Raw text the LLM returned.
        errors: Parse/validation error message from the last attempt.
        attempts: Number of attempts made before giving up.
    """

    def __init__(
        self,
        *,
        schema: dict | None = None,
        content: str = "",
        errors: str = "",
        attempts: int = 0,
    ):
        self.schema = schema
        self.content = content
        self.errors = errors
        self.attempts = attempts
        message = f"LLM output failed structured validation after {attempts} attempt(s): {errors}"
        super().__init__(message)


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
        json_schema: JSON Schema dict describing the expected response.
            When set, the response is parsed as JSON, validated against
            the schema, and re-asked (with the validation error fed back)
            up to *max_retries* times.  The parsed object is stored under
            *output_key*.  Adds ``response_format: {"type": "json_object"}``
            for OpenAI-compatible providers (``format: "json"`` for Ollama)
            unless *response_format* is already set.
        output_type: Python type spec — a ``TypedDict``, dataclass, or
            ``dict[str, type]`` field map — converted to a JSON Schema.
            Alternative to *json_schema*.
        parse: If ``True`` without a schema, parse the response as a JSON
            object and store the dict under *output_key* (no validation).
        max_retries: How many times to re-ask after a validation failure.
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
        json_schema: dict | None = None,
        output_type: typing.Type[typing.Any] | None = None,
        parse: bool = False,
        max_retries: int = 2,
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
            "json_schema": json_schema,
            "output_type": output_type,
            "parse": parse,
            "max_retries": max_retries,
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
                tool_defs.append(tool_to_schema(t))

        has_tools = bool(tool_defs)
        output_key = cfg.get("output_key", "output")

        schema = self._resolve_schema(cfg)
        structured = schema is not None
        parse_only = bool(cfg.get("parse", False)) and not structured

        harness = Harness.from_config(cfg, default_provider=self.DEFAULT_PROVIDER)
        harness.on_llm = self._record_llm_cb(ctx, cfg, provider_key)

        if structured and not cfg.get("response_format"):
            if provider_key == "ollama":
                harness._body_extra["format"] = "json"
            else:
                harness._body_extra["response_format"] = {"type": "json_object"}

        graph_stream = getattr(ctx, "emit", None) is not None
        content: str | dict = ""
        if (
            (cfg.get("stream", False) or graph_stream)
            and not has_tools
            and not structured
        ):
            harness.on_token = self._token_sink(ctx, cfg, provider_key)
            content = (await harness.call(messages, stream=True)).content
        else:
            max_retries = int(cfg.get("max_retries", 2))
            rounds = harness.max_rounds if has_tools else 1
            if structured:
                rounds = max(rounds, max_retries + 1)
            attempts = 0
            for _round in range(rounds):
                reply = await harness.call(
                    messages,
                    tools=tool_defs or None,
                    content_path=cfg.get("response_path", ""),
                )
                content = reply.content
                msg = reply.message
                tool_calls = msg.get("tool_calls")

                if has_tools and not tool_calls and harness.parse_text_tool_calls:
                    parsed = _parse_text_tool_call(content)
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
                        msg = {**msg, "tool_calls": tool_calls}

                if has_tools and tool_calls:
                    messages.append(msg)
                    results = await execute_tool_calls(
                        tool_calls, ctx.tools, harness.tool_error_mode
                    )
                    for tc, res in zip(tool_calls, results):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": res,
                            }
                        )
                    continue

                if structured:
                    assert schema is not None
                    assert isinstance(content, str)
                    parsed_value, error = self._parse_structured(content, schema)
                    if error is None:
                        content = parsed_value
                        break
                    attempts += 1
                    await self._record_structured(ctx, cfg, schema, error, attempts)
                    if attempts > max_retries:
                        raise StructuredOutputError(
                            schema=schema,
                            content=content,
                            errors=error,
                            attempts=attempts,
                        )
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response failed JSON schema "
                                f"validation: {error}\n"
                                "Respond with a single JSON object conforming "
                                f"to this schema:\n{json.dumps(schema)}"
                            ),
                        }
                    )
                    continue

                if parse_only:
                    raw = content
                    assert isinstance(raw, str)
                    try:
                        parsed = parse_json_object(raw)
                    except ValueError as exc:
                        raise StructuredOutputError(
                            content=raw,
                            errors=str(exc),
                            attempts=1,
                        ) from exc
                    content = parsed
                break

        if isinstance(content, dict):
            return {output_key: content}
        return {output_key: content or ""}

    def _resolve_provider(self, cfg: dict) -> str:
        p = cfg.get("provider") or self.DEFAULT_PROVIDER
        if p:
            return p.lower()
        detected = cfg.get("model", "gpt-4").split("-")[0].split("/")[0]
        return detected.lower()

    def _record_llm_cb(self, ctx, cfg: dict, provider_key: str):
        """Build the ``on_llm`` callback recording usage + ``llm`` events."""

        async def record(
            provider: str, model: str, prompt: int, completion: int, duration: float
        ) -> None:
            tracer = getattr(ctx, "tracer", None)
            if tracer is not None:
                tracer.llm(provider, model, prompt, completion, duration)
            emit = getattr(ctx, "emit", None)
            if emit is not None:
                await emit(
                    StreamEvent(
                        "llm",
                        node_id=ctx.node_id,
                        node_type=ctx.node_type,
                        data={
                            "provider": provider,
                            "model": model,
                            "prompt_tokens": prompt,
                            "completion_tokens": completion,
                            "duration_ms": duration,
                        },
                    )
                )

        return record

    def _resolve_schema(self, cfg: dict) -> dict | None:
        """Return the JSON Schema for structured output, if configured."""
        if cfg.get("json_schema") is not None:
            return json_schema_from_type(cfg["json_schema"])
        if cfg.get("output_type") is not None:
            return json_schema_from_type(cfg["output_type"])
        return None

    def _parse_structured(
        self, content: str, schema: dict
    ) -> tuple[typing.Any, str | None]:
        """Parse *content* as JSON and validate it against *schema*.

        Returns:
            ``(value, None)`` on success, ``(None, error_message)`` otherwise.
        """
        try:
            value = parse_json_object(content)
        except ValueError as exc:
            return None, str(exc)
        errors = validate_json(value, schema)
        if errors:
            return None, "; ".join(errors)
        return value, None

    async def _record_structured(
        self, ctx, cfg: dict, schema: dict, errors: str, attempt: int
    ) -> None:
        """Record a structured-output validation failure."""
        tracer = getattr(ctx, "tracer", None)
        if tracer is not None:
            tracer.structured(ctx.node_id, ctx.node_type, errors, attempt)
        emit = getattr(ctx, "emit", None)
        if emit is not None:
            await emit(
                StreamEvent(
                    "structured",
                    node_id=ctx.node_id,
                    node_type=ctx.node_type,
                    data={"errors": errors, "attempt": attempt},
                )
            )

    def _token_sink(
        self, ctx, cfg: dict, provider_key: str
    ) -> typing.Callable[[str], typing.Any]:
        """Build the per-token callback for streaming.

        Forwards each token to the node's ``on_token`` config and, when
        running under ``graph.stream()``, emits a ``token`` :class:`StreamEvent`.
        """
        emit = getattr(ctx, "emit", None)
        on_token = cfg.get("on_token")

        async def sink(token: str) -> None:
            if on_token is not None:
                on_token(token)
            if emit is not None:
                await emit(
                    StreamEvent(
                        "token",
                        node_id=ctx.node_id,
                        node_type=ctx.node_type,
                        data={
                            "token": token,
                            "provider": provider_key,
                            "model": str(cfg.get("model", "")),
                        },
                    )
                )

        return sink

    @staticmethod
    def _tool_to_schema(tool: Tool) -> dict:
        """Alias of :func:`draf.harness.tool_to_schema` (backward compat)."""
        return tool_to_schema(tool)
