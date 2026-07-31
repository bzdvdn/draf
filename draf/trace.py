"""Run tracing and telemetry for graph workflows.

Constitution Principle IX: observability is mandatory.  ``RunTracer``
collects a structured, JSON-serialisable event log for a single
``graph.run()`` call — timeline, per-node latency, retries, checkpoint
activity, and LLM token usage — and folds it into a ``RunSummary``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


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


@dataclass
class TraceEvent:
    """A single observability event emitted during a graph run.

    Attributes:
        kind: Event type — ``run_start``, ``node_start``, ``node_end``,
            ``node_error``, ``edge``, ``checkpoint``, ``llm``, ``retry``,
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
        """Return a JSON-serialisable dict for this event."""
        return {
            "kind": self.kind,
            "timestamp": round(self.timestamp, 6),
            "node_id": self.node_id,
            "node_type": self.node_type,
            "duration_ms": (
                None if self.duration_ms is None else round(self.duration_ms, 3)
            ),
            **self.data,
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
        return RunSummary(
            status=str(end_data.get("status", "ok")),
            total_ms=float(end_data.get("total_ms", 0.0)),
            node_count=len(nodes),
            llm_calls=self._llm_calls,
            tokens=self._usage,
            nodes=nodes,
        )

    def to_json(self) -> str:
        """Return a JSON report: ``{summary, events}``."""
        return json.dumps(
            {"summary": asdict(self.summary()), "events": self.timeline()},
            indent=2,
        )
