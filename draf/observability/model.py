"""Observability data model for a graph run.

A :class:`Run` is the top-level unit an exporter persists: the run
metadata, the graph topology it executed, one span per visited node, and
one entry per LLM call — including the *full* request (the messages sent)
and response, not just token counts.  Every model has a ``to_dict()`` that
is JSON-serialisable, so exporters and the web UI share one shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphTopology:
    """A node/edge snapshot of the compiled graph (for visualisation)."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GraphTopology":
        return GraphTopology(
            nodes=list(data.get("nodes") or []),
            edges=list(data.get("edges") or []),
        )


@dataclass
class LLMCall:
    """One model call with the full request/response payload."""

    node_id: str | None
    provider: str
    model: str
    messages: list[dict[str, Any]]
    response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "provider": self.provider,
            "model": self.model,
            "messages": self.messages,
            "response": self.response,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "cached": self.cached,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "LLMCall":
        return LLMCall(
            node_id=data.get("node_id"),
            provider=str(data["provider"]),
            model=str(data["model"]),
            messages=list(data.get("messages") or []),
            response=str(data.get("response") or ""),
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            latency_ms=float(data.get("latency_ms") or 0.0),
            cached=bool(data.get("cached")),
        )


@dataclass
class ToolCall:
    """One tool invocation: what the model requested and what ran.

    Tool calls are parsed out of the LLM message payloads (assistant
    ``tool_calls`` blocks matched to the following ``role: tool`` results),
    so a node's tool usage is a first-class citizen, not buried in the raw
    messages.  ``ok`` is ``False`` when the tool returned an ``"Error: ..."``
    result.
    """

    name: str
    args: str = "{}"
    result: str = ""
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "result": self.result,
            "ok": self.ok,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ToolCall":
        return ToolCall(
            name=str(data.get("name") or ""),
            args=str(data.get("args") or "{}"),
            result=str(data.get("result") or ""),
            ok=bool(data.get("ok", True)),
        )


@dataclass
class SpanEvent:
    """One step of a node's execution, in chronological order.

    ``kind`` is ``"llm"`` or ``"tool"`` and ``index`` points into the
    span's ``llm_calls`` / ``tool_calls`` lists.  Together they let a UI
    render the exact sequence a node followed — LLM call, tool call and
    its result, next LLM call, and so on — instead of two separate piles.
    """

    kind: str
    index: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "index": self.index}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SpanEvent":
        return SpanEvent(
            kind=str(data.get("kind") or ""),
            index=int(data.get("index") or 0),
        )


@dataclass
class NodeSpan:
    """One node execution: timing, outcome, its LLM calls and tool calls."""

    node_id: str
    node_type: str
    start_ms: float
    end_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    events: list[SpanEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_ms or self.start_ms) - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "start_ms": round(self.start_ms, 3),
            "end_ms": None if self.end_ms is None else round(self.end_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error": self.error,
            "llm_calls": [call.to_dict() for call in self.llm_calls],
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "events": [event.to_dict() for event in self.events],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "NodeSpan":
        end = data.get("end_ms")
        return NodeSpan(
            node_id=str(data["node_id"]),
            node_type=str(data.get("node_type") or ""),
            start_ms=float(data.get("start_ms") or 0.0),
            end_ms=None if end is None else float(end),
            status=str(data.get("status") or "ok"),
            error=data.get("error"),
            llm_calls=[
                LLMCall.from_dict(call) for call in (data.get("llm_calls") or [])
            ],
            tool_calls=[
                ToolCall.from_dict(call) for call in (data.get("tool_calls") or [])
            ],
            events=[SpanEvent.from_dict(event) for event in (data.get("events") or [])],
        )


@dataclass
class Run:
    """A single executed run, ready to export or serve over the API."""

    name: str
    status: str
    total_ms: float
    owner: str | None = None
    checkpoint_id: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    created_at: float | None = None
    topology: GraphTopology = field(default_factory=GraphTopology)
    nodes: list[NodeSpan] = field(default_factory=list)

    @property
    def llm_calls(self) -> list[LLMCall]:
        calls: list[LLMCall] = []
        for node in self.nodes:
            calls.extend(node.llm_calls)
        return calls

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.llm_calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.llm_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "total_ms": round(self.total_ms, 3),
            "owner": self.owner,
            "checkpoint_id": self.checkpoint_id,
            "tags": self.tags,
            "notes": self.notes,
            "created_at": self.created_at,
            "topology": self.topology.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "llm_calls": [call.to_dict() for call in self.llm_calls],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Run":
        created_at = data.get("created_at")
        return Run(
            name=str(data["name"]),
            status=str(data.get("status") or "ok"),
            total_ms=float(data.get("total_ms") or 0.0),
            owner=data.get("owner"),
            checkpoint_id=data.get("checkpoint_id"),
            tags=list(data.get("tags") or []),
            notes=str(data.get("notes") or ""),
            created_at=None if created_at is None else float(created_at),
            topology=GraphTopology.from_dict(data.get("topology") or {}),
            nodes=[NodeSpan.from_dict(node) for node in (data.get("nodes") or [])],
        )
