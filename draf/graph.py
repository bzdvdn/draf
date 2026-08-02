"""Graph data structure for representing agent workflows."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from draf.node.node import Node
from draf.node.registry import NodeRegistry, default_registry
from draf.node.context import ExecContext
from draf.node.interrupt import GraphInterrupt
from draf.errors import WorkflowError
from draf.tool.tool import Tool
from draf.state import Reducer, State, apply_reducers
from draf.checkpoint import DEFAULT_OWNER, Checkpoint, Checkpointer
from draf.trace import RunTracer, _ms
from draf.stream import StreamEvent


_ERROR_CONDITION = "__error__"
_INTERRUPT_KEY = "__interrupt__"
_MISSING = object()


@dataclass
class Edge:
    """A directed edge between two nodes with an optional condition.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        condition: Expression ``key=value``, ``key!=value``,
            or comma-separated disjunction ``key=a,b``.
            ``None`` means unconditional.
            ``"__error__"`` matches when the source node raises an exception.
    """

    source_id: str
    target_id: str
    condition: str | None = None


Hook = Callable[[str, Node, dict], Any]
"""Signature for observability hooks: ``(node_id, node, state)``."""


class Graph:
    """A directed graph of nodes connected by edges with conditions.

    The graph executes by walking from the *entry_point* node,
    following edges whose conditions match the current state,
    and shallow-merging each node's output back into the state.

    Error handling::

        Edge("parse", "fallback", "__error__")   # catch exceptions

    Observability hooks::

        await graph.run(state, hooks={
            "on_node_start": callback,
            "on_node_end": callback,
            "on_node_error": callback,
        })

    Hook callbacks receive ``(node_id, node, state)``.
    ``on_node_end`` additionally receives the result dict.
    ``on_node_error`` additionally receives the exception.
    """

    def __init__(self, nodes: dict[str, Node], edges: list[Edge], entry_point: str):
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point

    async def run(
        self,
        state: dict | State,
        tools: list[Tool] | None = None,
        registry: NodeRegistry | None = None,
        reducers: dict[str, Reducer] | None = None,
        hooks: dict[str, Callable] | None = None,
        node_timeout: float | None = None,
        max_iterations: int | None = None,
        checkpointer: Checkpointer | None = None,
        checkpoint_id: str | None = None,
        owner: str = DEFAULT_OWNER,
        resume: dict | None = None,
        tracer: RunTracer | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
    ) -> dict | State:
        """Execute the graph starting from the entry point.

        Args:
            state: Initial workflow state (plain ``dict`` or :class:`State`).
            tools: Optional list of Tool instances available to nodes.
            registry: Node registry (defaults to ``default_registry``).
            reducers: Per-key merge strategies
                (see :func:`draf.state.reducers_from_typeddict`).
                Ignored when *state* is a :class:`State` instance.
            hooks: Observability hooks (see class docstring).
            node_timeout: Max seconds per node.  ``asyncio.TimeoutError``
                triggers error edges (``__error__``) like any other exception.
            max_iterations: Max total node executions before raising
                ``RuntimeError``.  Guards against infinite loops in
                cyclic graphs (e.g. agentic loops).  ``None`` means unlimited.
            checkpointer: Optional persistence backend.  When set, a
                checkpoint is written before each node execution, so a
                crashed or interrupted run can be resumed by calling
                ``run`` again with the same *checkpoint_id* and the new
                initial state ignored in favor of the saved one.
            checkpoint_id: Key identifying a run (e.g. ``"thread-1"``).
                Required when *checkpointer* is set.  On a fresh ID the
                graph starts from *state* at the entry point; on an
                existing ID it resumes from the saved checkpoint.
            owner: Scopes *checkpoint_id* to a user/session/tenant.  The
                same ID under different owners never collides, and
                ``checkpointer.list(owner)`` enumerates a user's runs.
                Use one owner per end-user so every tenant's conversations
                stay isolated.  Defaults to
                :data:`draf.checkpoint.DEFAULT_OWNER`.
            resume: When a :class:`~draf.node.interrupt.Interrupt` node
                paused the run, pass a dict of ``{key: value}`` answers.
                Each key is written into the state before execution
                continues past the interrupt.  ``None`` on a normal run.
            tracer: Optional :class:`~draf.trace.RunTracer` collecting
                an event log for this run — timeline, node latency,
                retries, checkpoint activity, and LLM token usage.
                Inspect ``tracer.events`` / ``tracer.summary()`` after
                the run completes.
            emit: Optional async sink receiving
                :class:`~draf.stream.StreamEvent` objects as the run
                progresses.  Behaves like :meth:`stream` (emitting a
                final ``run_end`` event) but returns the final state
                instead of yielding events; used by nodes such as
                :class:`~draf.flow.sub_flow.SubFlow` to forward nested
                events, or for programmatic streaming.

        Raises:
            RuntimeError: If *max_iterations* is exceeded.
            GraphInterrupt: When an ``Interrupt`` node is reached.  The
                exception carries ``key``/``prompt`` for the operator;
                resume by calling ``run`` again with the same
                *checkpoint_id* and a ``resume`` dict.

        Returns:
            Final state (same type as passed in).
        """
        started = time.monotonic()
        try:
            result = await self._execute(
                state,
                tools=tools,
                registry=registry,
                reducers=reducers,
                hooks=hooks,
                node_timeout=node_timeout,
                max_iterations=max_iterations,
                checkpointer=checkpointer,
                checkpoint_id=checkpoint_id,
                owner=owner,
                resume=resume,
                tracer=tracer,
                emit=emit,
            )
        except GraphInterrupt:
            raise
        except Exception as exc:
            if tracer is not None:
                tracer.run_end("error", _ms(started), exc)
            if emit is not None:
                await emit(
                    StreamEvent(
                        "run_end", data={"status": "error", "error": str(exc)}
                    )
                )
            raise
        if tracer is not None:
            tracer.run_end("ok", _ms(started))
        if emit is not None:
            await emit(
                StreamEvent("run_end", data={"status": "ok", "total_ms": _ms(started)})
            )
        return result

    async def stream(
        self,
        state: dict | State,
        tools: list[Tool] | None = None,
        registry: NodeRegistry | None = None,
        reducers: dict[str, Reducer] | None = None,
        hooks: dict[str, Callable] | None = None,
        node_timeout: float | None = None,
        max_iterations: int | None = None,
        checkpointer: Checkpointer | None = None,
        checkpoint_id: str | None = None,
        owner: str = DEFAULT_OWNER,
        resume: dict | None = None,
        tracer: RunTracer | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events as the graph executes.

        Behaves like :meth:`run` but yields a :class:`StreamEvent` for
        each observable step instead of returning a final state::

            async for event in graph.stream(state):
                if event.type == "token":
                    print(event.data["token"], end="", flush=True)

        Event types: ``run_start``, ``node_start``, ``node_end``,
        ``node_error``, ``edge``, ``token``, ``llm``, ``structured``,
        ``interrupt``, ``interrupt_resume``, ``checkpoint``, and a final
        ``run_end``.
        Token events are only emitted when the running LLM node streams
        (any node with no tool calls streams automatically in this mode).

        A run paused at an ``Interrupt`` node ends with an ``interrupt``
        event (no ``run_end``); call :meth:`stream` or :meth:`run` again
        with a ``resume`` dict and the same *checkpoint_id* to continue.
        A failed run yields a ``run_end`` event with ``status: "error"``.

        Parameters mirror :meth:`run` (including ``owner``).
        """
        if checkpointer is not None and checkpoint_id is None:
            raise ValueError("checkpoint_id is required when checkpointer is set")
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        started = time.monotonic()

        async def _emit(event: StreamEvent) -> None:
            await queue.put(event)

        async def _runner() -> None:
            try:
                try:
                    await self._execute(
                        state,
                        tools=tools,
                        registry=registry,
                        reducers=reducers,
                        hooks=hooks,
                        node_timeout=node_timeout,
                        max_iterations=max_iterations,
                        checkpointer=checkpointer,
                        checkpoint_id=checkpoint_id,
                        owner=owner,
                        resume=resume,
                        tracer=tracer,
                        emit=_emit,
                    )
                except GraphInterrupt:
                    return
                except Exception as exc:
                    if tracer is not None:
                        tracer.run_end("error", _ms(started), exc)
                    await _emit(
                        StreamEvent(
                            "run_end", data={"status": "error", "error": str(exc)}
                        )
                    )
                    return
                if tracer is not None:
                    tracer.run_end("ok", _ms(started))
                await _emit(
                    StreamEvent(
                        "run_end", data={"status": "ok", "total_ms": _ms(started)}
                    )
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(_runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _execute(
        self,
        state: dict | State,
        tools: list[Tool] | None = None,
        registry: NodeRegistry | None = None,
        reducers: dict[str, Reducer] | None = None,
        hooks: dict[str, Callable] | None = None,
        node_timeout: float | None = None,
        max_iterations: int | None = None,
        checkpointer: Checkpointer | None = None,
        checkpoint_id: str | None = None,
        owner: str = "",
        resume: dict | None = None,
        tracer: RunTracer | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
    ) -> dict | State:
        """Shared execution core for :meth:`run` and :meth:`stream`.

        Returns the final state on success and raises on failure
        (``GraphInterrupt`` to pause, other exceptions for errors).
        ``run_end`` bookkeeping is left to the caller.
        """
        if checkpointer is not None:
            if checkpoint_id is None:
                raise ValueError("checkpoint_id is required when checkpointer is set")
            cid: str = checkpoint_id
        else:
            cid = ""

        registry = registry or default_registry
        tool_dict: dict[str, Tool] = {}
        if tools:
            for t in tools:
                tool_dict[t.name] = t

        hooks = hooks or {}

        if isinstance(state, State):
            reducers = reducers or state.reducers
        else:
            reducers = reducers or {}

        current_id: str | None = self.entry_point
        iteration = 0

        if tracer is not None:
            tracer.run_start(checkpoint_id=cid or None)
        if emit is not None:
            await emit(StreamEvent("run_start", data={"checkpoint_id": cid or None}))

        pending: dict | None = None
        if checkpointer is not None:
            saved = await checkpointer.load(cid, owner=owner)
            if tracer is not None:
                tracer.checkpoint(
                    "load",
                    cid,
                    saved.next_node_id if saved is not None else None,
                )
            if emit is not None:
                await emit(
                    StreamEvent(
                        "checkpoint",
                        data={
                            "action": "load",
                            "checkpoint_id": cid,
                            "next_node_id": (
                                saved.next_node_id if saved is not None else None
                            ),
                        },
                    )
                )
            if saved is not None:
                current_id = saved.next_node_id
                iteration = saved.iteration
                state = _restore_state(state, saved.state)
                pending = (
                    state.pop(_INTERRUPT_KEY, None) if isinstance(state, dict) else None
                )

        if pending is not None:
            if not resume:
                raise GraphInterrupt(
                    key=pending["key"],
                    prompt=pending.get("prompt", ""),
                    node_id=pending.get("node_id"),
                    checkpoint_id=cid or None,
                )
            for key, value in resume.items():
                state[key] = value
            if tracer is not None:
                tracer.interrupt_resume(pending.get("node_id"), list(resume.keys()))
            if emit is not None:
                await emit(
                    StreamEvent(
                        "interrupt_resume",
                        node_id=pending.get("node_id"),
                        data={"keys": list(resume.keys())},
                    )
                )
            interrupt_node = pending.get("node_id")
            if interrupt_node is not None:
                outgoing = [
                    e
                    for e in self.edges
                    if e.source_id == interrupt_node and e.condition != _ERROR_CONDITION
                ]
                current_id = self._resolve_edge(outgoing, state) if outgoing else None

        while current_id:
            if max_iterations is not None and iteration >= max_iterations:
                raise WorkflowError(f"graph exceeded max_iterations={max_iterations}")
            iteration += 1

            node = self.nodes[current_id]
            ctx = ExecContext(
                state,
                tool_dict,
                node_id=current_id,
                node_type=node.type,
                tracer=tracer,
                reducers=reducers,
                emit=emit,
            )
            start = time.monotonic()

            if checkpointer is not None:
                await checkpointer.save(
                    cid,
                    Checkpoint(
                        state=dict(state),
                        next_node_id=current_id,
                        iteration=iteration - 1,
                    ),
                    owner=owner,
                )
                if tracer is not None:
                    tracer.checkpoint("save", cid, current_id)
                if emit is not None:
                    await emit(
                        StreamEvent(
                            "checkpoint",
                            data={
                                "action": "save",
                                "checkpoint_id": cid,
                                "next_node_id": current_id,
                            },
                        )
                    )

            _call_hook(hooks, "on_node_start", current_id, node, state)
            if tracer is not None:
                tracer.node_start(current_id, node.type)
            if emit is not None:
                await emit(
                    StreamEvent("node_start", node_id=current_id, node_type=node.type)
                )

            try:
                if node_timeout is not None:
                    result = await asyncio.wait_for(
                        node.execute(ctx, state), timeout=node_timeout
                    )
                else:
                    result = await node.execute(ctx, state)
            except GraphInterrupt as exc:
                if tracer is not None:
                    tracer.interrupt(current_id, exc.key, exc.prompt)
                if emit is not None:
                    await emit(
                        StreamEvent(
                            "interrupt",
                            node_id=current_id,
                            data={"key": exc.key, "prompt": exc.prompt},
                        )
                    )
                if checkpointer is not None:
                    pending = dict(state)
                    pending[_INTERRUPT_KEY] = {
                        "key": exc.key,
                        "prompt": exc.prompt,
                        "node_id": current_id,
                    }
                    await checkpointer.save(
                        cid,
                        Checkpoint(
                            state=pending,
                            next_node_id=None,
                            iteration=iteration,
                        ),
                        owner=owner,
                    )
                    if tracer is not None:
                        tracer.checkpoint("save", cid, None)
                    if emit is not None:
                        await emit(
                            StreamEvent(
                                "checkpoint",
                                data={
                                    "action": "save",
                                    "checkpoint_id": cid,
                                    "next_node_id": None,
                                },
                            )
                        )
                exc.node_id = current_id
                exc.checkpoint_id = cid or None
                raise
            except Exception as exc:
                if tracer is not None:
                    tracer.node_error(current_id, node.type, _ms(start), exc)
                _call_hook(hooks, "on_node_error", current_id, node, state, exc)
                if emit is not None:
                    await emit(
                        StreamEvent(
                            "node_error",
                            node_id=current_id,
                            node_type=node.type,
                            data={"error": str(exc), "duration_ms": _ms(start)},
                        )
                    )
                error_edge = self._find_error_edge(current_id)
                if error_edge is not None:
                    current_id = error_edge.target_id
                    if checkpointer is not None:
                        await checkpointer.save(
                            cid,
                            Checkpoint(
                                state=dict(state),
                                next_node_id=current_id,
                                iteration=iteration,
                            ),
                            owner=owner,
                        )
                        if tracer is not None:
                            tracer.checkpoint("save", cid, current_id)
                        if emit is not None:
                            await emit(
                                StreamEvent(
                                    "checkpoint",
                                    data={
                                        "action": "save",
                                        "checkpoint_id": cid,
                                        "next_node_id": current_id,
                                    },
                                )
                            )
                    continue
                raise

            _call_hook(hooks, "on_node_end", current_id, node, state, result)
            if tracer is not None:
                tracer.node_end(current_id, node.type, _ms(start))
            if emit is not None:
                await emit(
                    StreamEvent(
                        "node_end",
                        node_id=current_id,
                        node_type=node.type,
                        data={"duration_ms": _ms(start)},
                    )
                )

            if result:
                if isinstance(state, State):
                    state.merge(result)
                else:
                    apply_reducers(state, result, reducers or {})

            outgoing = [
                e
                for e in self.edges
                if e.source_id == current_id and e.condition != _ERROR_CONDITION
            ]
            if not outgoing:
                break

            next_id = self._resolve_edge(outgoing, state)
            if next_id is None:
                break
            if tracer is not None:
                tracer.edge(
                    current_id,
                    next_id,
                    self._matched_condition(outgoing, state, next_id),
                )
            if emit is not None:
                await emit(
                    StreamEvent(
                        "edge",
                        node_id=current_id,
                        data={
                            "target_id": next_id,
                            "condition": self._matched_condition(
                                outgoing, state, next_id
                            ),
                        },
                    )
                )
            current_id = next_id

        if checkpointer is not None:
            await checkpointer.save(
                cid,
                Checkpoint(state=dict(state), next_node_id=None, iteration=iteration),
                owner=owner,
            )
            if tracer is not None:
                tracer.checkpoint("save", cid, None)
            if emit is not None:
                await emit(
                    StreamEvent(
                        "checkpoint",
                        data={
                            "action": "save",
                            "checkpoint_id": cid,
                            "next_node_id": None,
                        },
                    )
                )

        return state

    def _find_error_edge(self, node_id: str) -> Edge | None:
        for e in self.edges:
            if e.source_id == node_id and e.condition == _ERROR_CONDITION:
                return e
        return None

    def _resolve_edge(self, edges: list[Edge], state: dict) -> str | None:
        for edge in edges:
            if edge.condition is None:
                return edge.target_id
            if self._evaluate(edge.condition, state):
                return edge.target_id
        return None

    def _matched_condition(
        self, edges: list[Edge], state: dict, target_id: str
    ) -> str | None:
        """Return the condition of the first edge matching *state* and *target_id*."""
        for edge in edges:
            if edge.target_id != target_id:
                continue
            if edge.condition is None or self._evaluate(edge.condition, state):
                return edge.condition
        return None

    @staticmethod
    def _norm(value: str) -> str:
        """Normalise a condition value for comparison.

        Strips whitespace, lowercases, and removes trailing punctuation
        so LLM output like ``"Positive."`` matches ``positive``.
        """
        return value.strip().lower().rstrip(".,!?;:")

    def _evaluate(self, condition: str, state: dict) -> bool:
        if "!=" in condition:
            key, value = condition.split("!=", 1)
            key = key.strip()
            raw = value.strip()
            state_val = state.get(key)
            if raw == "":
                return state_val is not None and state_val != ""
            if state_val is None:
                return True
            state_str = self._norm(str(state_val))
            if "," in raw:
                values = [self._norm(v) for v in raw.split(",")]
                return state_str not in values
            return state_str != self._norm(raw)
        if "=" in condition:
            parts = condition.split("=", 1)
            key = parts[0].strip()
            raw = parts[1].strip()
            state_val = state.get(key)
            if raw == "":
                return state_val is None or state_val == ""
            if state_val is None:
                return False
            state_str = self._norm(str(state_val))
            if "," in raw:
                values = [self._norm(v) for v in raw.split(",")]
                return state_str in values
            return state_str == self._norm(raw)
        return False

    def to_yaml(self) -> str:
        """Serialize this graph to a YAML string."""
        from draf.yaml import graph_to_yaml

        return graph_to_yaml(self)


def _call_hook(hooks: dict, name: str, *args: Any) -> None:
    fn = hooks.get(name)
    if fn is not None:
        fn(*args)


def _restore_state(original: dict | State, data: dict) -> dict | State:
    """Rehydrate state from checkpoint data, preserving a State wrapper.

    If *original* is a :class:`State` instance its schema (and reducers)
    are kept and only the data is replaced. Plain dicts are returned as-is.
    """
    if isinstance(original, State):
        original.clear()
        original.update(data)
        return original
    return data
