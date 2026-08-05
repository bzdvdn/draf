"""Graph data structure for representing agent workflows."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable

from draf.checkpoint import DEFAULT_OWNER, Checkpointer
from draf.graph.edge import Edge, Hook
from draf.graph.execution import execute
from draf.graph.render import to_mermaid
from draf.node.interrupt import GraphInterrupt
from draf.node.node import Node
from draf.node.registry import NodeRegistry
from draf.provider import (
    Provider,
    ProviderRegistry,
    to_provider_registry,
)
from draf.state import Reducer, State
from draf.stream import StreamEvent
from draf.tool.tool import Tool
from draf.trace import RunTracer, _ms

__all__ = ["Edge", "Graph", "Hook"]


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
    ``on_node_end`` additionally receives the result dict and runs *after*
    the result is merged into state (so it observes the node's effect).
    ``on_node_error`` additionally receives the exception.
    Hooks may be sync or async; async hooks are awaited.
    """

    def __init__(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
        entry_point: str,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.providers: ProviderRegistry = to_provider_registry(providers)
        self.default_provider: str | None = default_provider
        self.default_model: str | None = default_model

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
        state_schema: dict | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
    ) -> dict | State:
        """Execute the graph starting from the entry point.

        Args:
            state: Initial workflow state (plain ``dict`` or :class:`State`).
            tools: Optional list of Tool instances available to nodes.
            providers: Optional ``{name: Provider}`` map or
                :class:`~draf.provider.ProviderRegistry` consulted by LLM
                nodes before the built-in presets.  Defaults to
                ``graph.providers`` (populated from a workflow's
                ``providers:`` block when loaded from YAML).
            default_provider: Optional default provider name used by LLM
                nodes that don't set ``provider`` themselves.  Defaults to
                ``graph.default_provider`` (``Graph(default_provider=...)``
                or a workflow's top-level ``default_provider:``).
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
            state_schema: Optional YAML ``state.schema`` dict.  When set,
                *state* is validated against it before execution and a
                :class:`~draf.errors.ConfigError` is raised on mismatch.
                See :func:`draf.state.validate_state`.

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
            result = await execute(
                self,
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
                state_schema=state_schema,
                emit=emit,
                providers=providers if providers is not None else self.providers,
                default_provider=default_provider
                if default_provider is not None
                else self.default_provider,
                default_model=default_model
                if default_model is not None
                else self.default_model,
                on_llm_payload=on_llm_payload,
            )
        except GraphInterrupt:
            raise
        except Exception as exc:
            if tracer is not None:
                tracer.run_end("error", _ms(started), exc)
            if emit is not None:
                await emit(
                    StreamEvent("run_end", data={"status": "error", "error": str(exc)})
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
        state_schema: dict | None = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
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
        queue: "asyncio.Queue[StreamEvent | None]" = asyncio.Queue()
        started = time.monotonic()

        async def _emit(event: StreamEvent) -> None:
            await queue.put(event)

        async def _runner() -> None:
            try:
                try:
                    await execute(
                        self,
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
                        state_schema=state_schema,
                        emit=_emit,
                        providers=providers
                        if providers is not None
                        else self.providers,
                        default_provider=default_provider
                        if default_provider is not None
                        else self.default_provider,
                        default_model=default_model
                        if default_model is not None
                        else self.default_model,
                        on_llm_payload=on_llm_payload,
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

    def to_yaml(self) -> str:
        """Serialize this graph to a YAML string."""
        from draf.yaml import graph_to_yaml

        return graph_to_yaml(self)

    def to_mermaid(self, show_conditions: bool = True) -> str:
        """Render this graph as a Mermaid flowchart diagram.

        Produces a ``flowchart TD`` definition: every node becomes a box
        labelled ``node_id[node.type]`` and every edge an arrow.  The entry
        point is filled blue, ``__error__`` edges are dashed and red, and
        conditional edges carry their condition as an edge label (when
        *show_conditions* is true).

        Returns:
            The Mermaid diagram as a string (no code fence).
        """
        return to_mermaid(self, show_conditions=show_conditions)
