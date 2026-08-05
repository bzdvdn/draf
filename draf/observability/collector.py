"""Collect a full graph-run trace into a single :class:`Run`.

:class:`GraphObserver` is the wiring point between ``graph.run()`` and the
exporter layer.  Attach it by passing its three channels to the run::

    observer = GraphObserver(
        "repair-agent", exporter=SQLiteExporter("traces.db"),
        topology=topology_from_graph(graph),
    )
    state = await graph.run(
        state,
        owner=owner,
        tracer=observer.tracer,              # node/edge/checkpoint events
        on_llm_payload=observer.on_llm_payload,  # full prompt/response
    )
    observer.export()

It reuses :class:`~draf.trace.RunTracer` for the structural events and adds
the LLM payload hook that the run already forwards to every harness, so the
collected :class:`Run` contains the complete graph picture — topology,
per-node spans, and every model call with its messages.
"""

from __future__ import annotations

import time
from typing import Any

from draf.observability.exporter import TraceExporter
from draf.observability.model import GraphTopology, LLMCall, NodeSpan, Run
from draf.trace import RunTracer


class GraphObserver:
    """Assemble a :class:`Run` from graph events and forward it to an exporter."""

    def __init__(
        self,
        name: str,
        *,
        exporter: TraceExporter | None = None,
        topology: GraphTopology | None = None,
        owner: str | None = None,
        checkpoint_id: str | None = None,
    ):
        self.name = name
        self.exporter = exporter
        self.topology = topology or GraphTopology()
        self.owner = owner
        self.checkpoint_id = checkpoint_id

        self.tracer = RunTracer()
        self._start = time.monotonic()
        self._wall_start = time.time()
        self._spans: dict[str, NodeSpan] = {}
        self._active: list[NodeSpan] = []
        self._status = "ok"
        self._error: str | None = None
        self._total_ms = 0.0
        self._wire_tracer()

    async def on_llm_payload(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        response: str,
        usage: dict[str, Any],
        latency_ms: float,
        cached: bool,
    ) -> None:
        """Sink for the run's ``on_llm_payload`` channel."""
        call = LLMCall(
            node_id=self._active[-1].node_id if self._active else None,
            provider=provider,
            model=model,
            messages=messages,
            response=response,
            prompt_tokens=int(usage.get("prompt", 0) or 0),
            completion_tokens=int(usage.get("completion", 0) or 0),
            latency_ms=latency_ms,
            cached=cached,
        )
        if self._active:
            self._active[-1].llm_calls.append(call)

    def _start_node(self, node_id: str, node_type: str) -> None:
        span = NodeSpan(
            node_id=node_id,
            node_type=node_type,
            start_ms=(time.monotonic() - self._start) * 1000.0,
        )
        self._spans[node_id] = span
        self._active.append(span)

    def _end_node(
        self, node_id: str, status: str = "ok", error: str | None = None
    ) -> None:
        span = self._spans.get(node_id)
        if span is None:
            return
        span.end_ms = (time.monotonic() - self._start) * 1000.0
        span.status = status
        span.error = error
        if self._active and self._active[-1] is span:
            self._active.pop()

    def _wire_tracer(self) -> None:
        tracer = self.tracer
        node_start = tracer.node_start
        node_end = tracer.node_end
        node_error = tracer.node_error
        run_end = tracer.run_end

        def _node_start(node_id, node_type):
            node_start(node_id, node_type)
            self._start_node(node_id, node_type)

        def _node_end(node_id, node_type, duration_ms):
            node_end(node_id, node_type, duration_ms)
            self._end_node(node_id)

        def _node_error(node_id, node_type, duration_ms, error):
            node_error(node_id, node_type, duration_ms, error)
            self._end_node(node_id, status="error", error=str(error))

        def _run_end(status, total_ms, error=None):
            run_end(status, total_ms, error)
            self._status = status
            self._total_ms = total_ms
            if error is not None:
                self._error = str(error)
            for span in list(self._active):
                self._end_node(span.node_id)

        tracer.node_start = _node_start  # type: ignore[method-assign]
        tracer.node_end = _node_end  # type: ignore[method-assign]
        tracer.node_error = _node_error  # type: ignore[method-assign]
        tracer.run_end = _run_end  # type: ignore[method-assign]

    def build(self) -> Run:
        return Run(
            name=self.name,
            status=self._status,
            total_ms=self._total_ms,
            owner=self.owner,
            checkpoint_id=self.checkpoint_id,
            created_at=self._wall_start,
            topology=self.topology,
            nodes=list(self._spans.values()),
        )

    def export(self) -> None:
        """Persist the collected :class:`Run` (no-op without an exporter)."""
        if self.exporter is not None:
            self.exporter.export(self.build())

    def close(self) -> None:
        if self.exporter is not None:
            self.exporter.close()
