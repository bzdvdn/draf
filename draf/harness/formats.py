"""Response parsing and message-format normalisation for LLM providers."""

from __future__ import annotations

import json
import re
import typing

from draf.schema import extract_json_object


def extract_message(data: dict) -> dict:
    """Normalise response formats to ``{role, content, tool_calls}``.

    Handles OpenAI (``data["choices"][0]["message"]``) and
    Ollama (``data["message"]`` at root).
    """
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})
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
                        "arguments": json.dumps(block.get("input", {}), default=str),
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


def normalize_text_tool_calls(
    content: str, msg: dict, *, seq: int = 0
) -> tuple[list[dict], dict]:
    """Turn a text-embedded tool call into the structured ``tool_calls`` list.

    When ``content`` parses as a single ``{name, arguments|parameters}``
    object, returns ``([tool_call], msg_with_tool_calls)``; otherwise
    returns ``([], msg)`` unchanged.  The generated ``call_id`` is derived
    from *seq* + the tool name so it is unique within a run.

    Returns:
        A ``(tool_calls, message)`` pair.  *message* is *msg* with
        ``tool_calls`` attached when a text call was found.
    """
    parsed = parse_text_tool_call(content)
    if not parsed:
        return [], msg
    name, args = parsed
    call_id = f"call_{seq}_{name}"
    tool_calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
    ]
    return tool_calls, {**msg, "tool_calls": tool_calls}


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
    data: dict, provider_type: str, path: str = "", fallback: str = ""
) -> str:
    """Extract the assistant text from a response.

    *path* is a dot-separated path into *data*; otherwise the extraction
    follows the wire protocol *provider_type* (Anthropic content blocks,
    Ollama root ``message``).
    """
    if path:
        parts = path.split(".")
        val: typing.Any = data
        try:
            for p in parts:
                if p.isdigit():
                    val = val[int(p)]
                else:
                    val = val.get(p, "")
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return ""
        return str(val) if val else ""

    if provider_type == "anthropic_compatible":
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "")
        return ""

    if provider_type == "ollama":
        return data.get("message", {}).get("content", "")

    return fallback


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
