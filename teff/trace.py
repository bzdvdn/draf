"""Run tracing and telemetry for graph workflows.

Constitution Principle IX: observability is mandatory.  ``RunTracer``
collects a structured, JSON-serialisable event log for a single
``graph.run()`` call — timeline, per-node latency, retries, checkpoint
activity, and LLM token usage — and folds it into a ``RunSummary``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

from teff.errors import redact

# Approximate list pricing in USD per 1M tokens (input, output).
# Local/self-hosted models are not in the table and cost $0.
# Used as a fallback when no provider-specific pricing is registered.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.5, 1.5),
    "o1": (15.0, 60.0),
    "o1-mini": (1.1, 4.4),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-sonnet": (3.0, 15.0),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-5-haiku": (0.8, 4.0),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 1.19),
    "mistral-large": (2.0, 6.0),
    "mistral-medium": (2.7, 8.1),
    "mistral-small": (0.2, 0.6),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.3),
}

# Provider-scoped pricing registered at runtime.  Custom providers such as
# OpenRouter or aggregators keep their own per-model rates and use their own
# model names (e.g. ``openai/gpt-4o``), so a model-only table cannot cover
# them.  Keys are ``(provider, model)``; values are ``(input, output)`` USD
# per 1M tokens.
_CUSTOM_PRICING: dict[tuple[str, str], tuple[float, float]] = {}
# Provider-wide defaults keyed by provider, applied when no per-model entry
# (custom or built-in) matches.
_PROVIDER_PRICING: dict[str, tuple[float, float]] = {}


def set_model_pricing(
    provider: str,
    model: str,
    input_price: float,
    output_price: float,
) -> None:
    """Register custom USD-per-1M-token pricing for a provider/model pair.

    Takes precedence over the built-in table and any provider-wide default.
    """
    _CUSTOM_PRICING[(provider.lower(), model)] = (
        float(input_price),
        float(output_price),
    )


def set_provider_pricing(
    provider: str,
    input_price: float,
    output_price: float,
) -> None:
    """Register a provider-wide default price in USD per 1M tokens.

    Applied to every model on *provider* that has no per-model entry.
    """
    _PROVIDER_PRICING[provider.lower()] = (float(input_price), float(output_price))


def load_pricing(source: str | dict) -> None:
    """Register pricing from a YAML/JSON file path or an inline dict.

    Format::

        providers:
          openrouter:
            default: {input: 0.1, output: 0.4}
            models:
              "openai/gpt-4o": {input: 3.0, output: 12.0}
              "anthropic/claude-3.5-sonnet": {input: 3.0, output: 15.0}

    A flat dict ``{"provider": {model: [in, out]}}`` is also accepted.
    """
    if isinstance(source, str) and os.path.exists(source):
        with open(source) as f:
            data = yaml.safe_load(f) or {}
        data = data.get("providers", data)
    elif isinstance(source, str):
        raise ValueError("load_pricing expects an existing file path or a dict")
    else:
        data = source

    for provider, block in data.items():
        if not isinstance(block, dict):
            continue
        default = block.get("default")
        if default:
            _PROVIDER_PRICING[provider.lower()] = (
                float(default["input"]),
                float(default["output"]),
            )
        for model, price in (block.get("models") or {}).items():
            if isinstance(price, (list, tuple)):
                _CUSTOM_PRICING[(provider.lower(), model)] = (
                    float(price[0]),
                    float(price[1]),
                )
            else:
                _CUSTOM_PRICING[(provider.lower(), model)] = (
                    float(price["input"]),
                    float(price["output"]),
                )


def clear_pricing() -> None:
    """Remove all custom pricing registered at runtime."""
    _CUSTOM_PRICING.clear()
    _PROVIDER_PRICING.clear()


def model_pricing(model: str, provider: str = "") -> tuple[float, float]:
    """Return ``(input, output)`` USD per 1M tokens for *model* / *provider*.

    Resolution order:

    1. exact ``(provider, model)`` custom entry;
    2. provider-prefixed custom entry (``gpt-4o`` matches ``gpt-4o-2024-08-06``);
    3. provider-wide default;
    4. built-in table (exact, then prefix);
    5. ``(0.0, 0.0)`` for unknown/local models.

    When *provider* is empty only the built-in table is consulted, so
    callers that pass no provider keep their current behaviour.
    """
    key = (provider.lower(), model)
    if key in _CUSTOM_PRICING:
        return _CUSTOM_PRICING[key]
    for (p, name), price in _CUSTOM_PRICING.items():
        if p == provider.lower() and model.startswith(name + "-"):
            return price
    if provider:
        p = provider.lower()
        if p in _PROVIDER_PRICING:
            return _PROVIDER_PRICING[p]
    if model in _MODEL_PRICING:
        return _MODEL_PRICING[model]
    for name, price in _MODEL_PRICING.items():
        if model.startswith(name + "-"):
            return price
    return (0.0, 0.0)


def tokens_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    provider: str = "",
) -> float:
    """Estimate the USD cost of a model call from its token usage."""
    input_price, output_price = model_pricing(model, provider)
    return (
        prompt_tokens / 1_000_000 * input_price
        + completion_tokens / 1_000_000 * output_price
    )


def _ms(start: float) -> float:
    """Elapsed milliseconds since *start* (a ``time.monotonic()`` stamp)."""
    return (time.monotonic() - start) * 1000.0


@dataclass
class TokenUsage:
    """Accumulated LLM token counts for a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class NodeStats:
    """Aggregated per-node statistics for a run."""

    runs: int = 0
    errors: int = 0
    total_ms: float = 0.0


@dataclass
class RunSummary:
    """Folded summary computed from a run's trace events."""

    status: str = "ok"
    total_ms: float = 0.0
    node_count: int = 0
    llm_calls: int = 0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    nodes: dict[str, NodeStats] = field(default_factory=dict)
    cost_usd: float = 0.0
    models: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (model names redacted)."""
        return {
            "status": self.status,
            "total_ms": round(self.total_ms, 3),
            "node_count": self.node_count,
            "llm_calls": self.llm_calls,
            "cost_usd": round(self.cost_usd, 6),
            "tokens": {
                "prompt_tokens": self.tokens.prompt_tokens,
                "completion_tokens": self.tokens.completion_tokens,
                "total": self.tokens.total,
            },
            "models": redact(self.models),
            "nodes": {
                nid: {
                    "runs": stats.runs,
                    "errors": stats.errors,
                    "total_ms": round(stats.total_ms, 3),
                }
                for nid, stats in sorted(self.nodes.items())
            },
        }

    def to_json(self) -> str:
        """Return this summary as a pretty-printed JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TraceEvent:
    """A single observability event emitted during a graph run.

    Attributes:
        kind: Event type — ``run_start``, ``node_start``, ``node_end``,
            ``node_error``, ``edge``, ``checkpoint``, ``llm``, ``retry``,
            ``structured``, ``interrupt``, ``interrupt_resume``,
            or ``run_end``.
        timestamp: Seconds since the tracer started (monotonic).
        node_id: Graph node id the event belongs to, if any.
        node_type: Node type string, if any.
        duration_ms: Node/LLM call duration in milliseconds, if measured.
        data: Kind-specific payload (error, condition, tokens, etc.).
    """

    kind: str
    timestamp: float
    node_id: str | None = None
    node_type: str | None = None
    duration_ms: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for this event (data redacted)."""
        return {
            "kind": self.kind,
            "timestamp": round(self.timestamp, 6),
            "node_id": self.node_id,
            "node_type": self.node_type,
            "duration_ms": (
                None if self.duration_ms is None else round(self.duration_ms, 3)
            ),
            **redact(self.data),
        }


class RunTracer:
    """Collects trace events during a ``graph.run()`` call.

    Pass an instance to ``graph.run(tracer=...)``.  After the run,
    inspect ``events`` for the raw timeline, ``timeline()`` for a
    JSON-serialisable list, ``summary()`` for aggregated statistics, or
    ``to_json()`` for a ready-to-persist report.

    Events are also emitted for the node-level hooks (start/end/error)
    plus edge routing, checkpoint saves/loads, retries, and LLM calls.
    """

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._start = time.monotonic()
        self._usage = TokenUsage()
        self._llm_calls = 0

    def _record(
        self,
        kind: str,
        node_id: str | None = None,
        node_type: str | None = None,
        duration_ms: float | None = None,
        **data: Any,
    ) -> None:
        self.events.append(
            TraceEvent(
                kind=kind,
                timestamp=time.monotonic() - self._start,
                node_id=node_id,
                node_type=node_type,
                duration_ms=duration_ms,
                data=data,
            )
        )

    def run_start(self, checkpoint_id: str | None = None) -> None:
        """Record the beginning of a run."""
        self._record("run_start", checkpoint_id=checkpoint_id)

    def node_start(self, node_id: str, node_type: str) -> None:
        """Record the start of a node execution."""
        self._record("node_start", node_id=node_id, node_type=node_type)

    def node_end(self, node_id: str, node_type: str, duration_ms: float) -> None:
        """Record the successful completion of a node."""
        self._record(
            "node_end",
            node_id=node_id,
            node_type=node_type,
            duration_ms=duration_ms,
        )

    def node_error(
        self,
        node_id: str,
        node_type: str,
        duration_ms: float,
        error: Exception,
    ) -> None:
        """Record a node failure."""
        self._record(
            "node_error",
            node_id=node_id,
            node_type=node_type,
            duration_ms=duration_ms,
            error=str(error),
        )

    def edge(
        self, source_id: str, target_id: str, condition: str | None = None
    ) -> None:
        """Record a routing decision from *source_id* to *target_id*."""
        self._record(
            "edge",
            node_id=source_id,
            target_id=target_id,
            condition=condition,
        )

    def checkpoint(
        self,
        action: str,
        checkpoint_id: str,
        next_node_id: str | None,
    ) -> None:
        """Record a checkpoint ``save`` or ``load``."""
        self._record(
            "checkpoint",
            checkpoint_id=checkpoint_id,
            action=action,
            next_node_id=next_node_id,
        )

    def llm(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
    ) -> None:
        """Record an LLM call and accumulate its token usage."""
        self._usage.prompt_tokens += prompt_tokens
        self._usage.completion_tokens += completion_tokens
        self._llm_calls += 1
        self._record(
            "llm",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
        )

    def retry(
        self,
        node_id: str | None,
        node_type: str | None,
        attempt: int,
        error: Exception,
    ) -> None:
        """Record a retry attempt (1-based attempt number)."""
        self._record(
            "retry",
            node_id=node_id,
            node_type=node_type,
            attempt=attempt,
            error=str(error),
        )

    def structured(
        self,
        node_id: str | None,
        node_type: str | None,
        errors: str,
        attempt: int,
    ) -> None:
        """Record a structured-output validation failure (1-based attempt)."""
        self._record(
            "structured",
            node_id=node_id,
            node_type=node_type,
            attempt=attempt,
            errors=errors,
        )

    def interrupt(self, node_id: str, key: str, prompt: str) -> None:
        """Record that an ``Interrupt`` node paused the run for input."""
        self._record(
            "interrupt",
            node_id=node_id,
            key=key,
            prompt=prompt,
        )

    def interrupt_resume(self, node_id: str | None, keys: list[str]) -> None:
        """Record that a paused run resumed with answers for *keys*."""
        self._record("interrupt_resume", node_id=node_id, keys=keys)

    def run_end(
        self,
        status: str,
        total_ms: float,
        error: Exception | None = None,
    ) -> None:
        """Record the end of a run (``status`` in ``{"ok", "error"}``)."""
        data: dict[str, Any] = {"status": status, "total_ms": total_ms}
        if error is not None:
            data["error"] = str(error)
        self._record("run_end", **data)

    def timeline(self) -> list[dict[str, Any]]:
        """Return the raw event log as JSON-serialisable dicts."""
        return [ev.to_dict() for ev in self.events]

    def summary(self) -> RunSummary:
        """Fold all events into an aggregated :class:`RunSummary`."""
        nodes: dict[str, NodeStats] = {}
        for ev in self.events:
            if ev.node_id is None:
                continue
            stats = nodes.setdefault(ev.node_id, NodeStats())
            if ev.kind == "node_start":
                stats.runs += 1
            elif ev.kind == "node_error":
                stats.errors += 1
            if ev.duration_ms is not None:
                stats.total_ms += ev.duration_ms

        run_end = next((e for e in reversed(self.events) if e.kind == "run_end"), None)
        end_data = run_end.data if run_end else {}

        cost = 0.0
        per_model: dict[str, dict[str, int]] = {}
        for ev in self.events:
            if ev.kind != "llm":
                continue
            model = str(ev.data.get("model", ""))
            provider = str(ev.data.get("provider", ""))
            prompt = int(ev.data.get("prompt_tokens", 0))
            completion = int(ev.data.get("completion_tokens", 0))
            cost += tokens_cost(model, prompt, completion, provider=provider)
            usage = per_model.setdefault(
                model, {"prompt_tokens": 0, "completion_tokens": 0}
            )
            usage["prompt_tokens"] += prompt
            usage["completion_tokens"] += completion

        return RunSummary(
            status=str(end_data.get("status", "ok")),
            total_ms=float(end_data.get("total_ms", 0.0)),
            node_count=len(nodes),
            llm_calls=self._llm_calls,
            tokens=self._usage,
            nodes=nodes,
            cost_usd=cost,
            models=per_model,
        )

    def to_json(self) -> str:
        """Return a JSON report: ``{summary, events}`` (secrets redacted)."""
        return json.dumps(
            {"summary": self.summary().to_dict(), "events": self.timeline()},
            indent=2,
        )
