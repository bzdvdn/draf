"""Durable conversation turns against a compiled graph.

:class:`Assistant` is the shared "one conversation turn" primitive: it
hydrates a session from the checkpointer (seeding a fresh one, or appending
the message to an existing session), runs the graph once, and can return
the latest assistant reply from the durable state.  Both the FastAPI server
and a terminal CLI go through it, so a turn behaves identically over HTTP
and on the terminal.

The state shape is supplied by the caller — the graph's reducers, the fresh
``initial_state``, which keys are per-turn transient scratch and the name of
the messages key — so the class is generic over any conversation graph
instead of being hard-wired to one application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Mapping

from draf.checkpoint import DEFAULT_OWNER, Checkpoint, Checkpointer
from draf.graph import Graph
from draf.graph.edge import _INTERRUPT_KEY
from draf.stream import StreamEvent
from draf.tool import Tool


@dataclass
class TurnResult:
    """Structured outcome of one :meth:`Assistant.turn`.

    Attributes:
        session_id: The session the turn ran against.
        reply: The latest assistant reply (``""`` when the turn is paused
            before any assistant text, e.g. ``waiting`` after a pure gate).
        waiting: ``True`` when the run paused on an
            :class:`~draf.node.Interrupt` and needs an operator answer.
        prompt: The interrupt's question (only when ``waiting``).
        key: The interrupt's state key (only when ``waiting``).
        state: Final state for a completed turn (``None`` when ``waiting``).
    """

    session_id: str
    reply: str = ""
    waiting: bool = False
    prompt: str | None = None
    key: str | None = None
    state: dict | None = None


class Assistant:
    """Runs durable conversation turns against a compiled graph."""

    def __init__(
        self,
        graph: Graph,
        tools: list[Tool],
        checkpointer: Checkpointer,
        *,
        reducers: dict | None = None,
        initial_state: Callable[[], Mapping[str, object]] | None = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
        max_iterations: int = 80,
    ):
        self.graph = graph
        self.tools = tools
        self.checkpointer = checkpointer
        self.reducers = reducers
        self.initial_state = initial_state
        self.transient_keys = transient_keys
        self.messages_key = messages_key
        self.max_iterations = max_iterations

    async def _load_or_seed(
        self,
        session_id: str,
        message: str,
        *,
        owner: str = DEFAULT_OWNER,
    ) -> tuple[dict, dict]:
        """Return ``(state, run_kwargs)`` for one conversation turn.

        * Fresh session  -> seed the state with the user message; ``graph.run``
          checkpoints as it executes.
        * Existing       -> append the message to the durable state and re-enter
          at the entry point so history drives the reply.  The *state* we return
          is empty because ``graph.run`` restores the just-saved checkpoint.
        """
        saved = await self.checkpointer.load(session_id, owner=owner)
        if saved is None:
            state: dict[str, Any] = dict(
                self.initial_state() if self.initial_state else {}
            )
            state[self.messages_key] = [{"role": "user", "content": message}]
            return state, {}

        state = dict(saved.state)
        messages = list(state.get(self.messages_key) or [])
        messages.append({"role": "user", "content": message})
        state[self.messages_key] = messages
        for key in self.transient_keys:
            state[key] = ""
        await self.checkpointer.save(
            session_id,
            Checkpoint(state=state, next_node_id=self.graph.entry_point, iteration=0),
            owner=owner,
        )
        return {}, {}

    async def run_turn(
        self,
        session_id: str,
        message: str,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer=None,
        on_llm_payload=None,
    ) -> dict:
        """Run one turn and return the final state."""
        state, run_kwargs = await self._load_or_seed(session_id, message, owner=owner)
        return await self.graph.run(
            state,
            tools=self.tools,
            reducers=self.reducers,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations or self.max_iterations,
            tracer=tracer,
            on_llm_payload=on_llm_payload,
            **run_kwargs,
        )

    async def resume_turn(
        self,
        session_id: str,
        resume: dict,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer=None,
        on_llm_payload=None,
    ) -> dict:
        """Resume a turn paused by an :class:`~draf.node.interrupt.Interrupt`.

        *resume* maps the interrupt's state key to the operator's answer,
        e.g. ``{"approved": "yes"}``.  ``graph.run`` restores the checkpoint
        saved when the interrupt fired and continues past it; a re-interrupt
        (e.g. a "rework" branch) raises again.
        """
        return await self.graph.run(
            {},
            tools=self.tools,
            reducers=self.reducers,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations or self.max_iterations,
            tracer=tracer,
            on_llm_payload=on_llm_payload,
            resume=resume,
        )

    async def stream_resume(
        self,
        session_id: str,
        resume: dict,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer=None,
        on_llm_payload=None,
    ) -> AsyncIterator[StreamEvent]:
        """Resume a paused turn (``stream_turn`` equivalent of :meth:`resume_turn`)."""
        async for event in self.graph.stream(
            state={},
            tools=self.tools,
            reducers=self.reducers,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations or self.max_iterations,
            tracer=tracer,
            on_llm_payload=on_llm_payload,
            resume=resume,
        ):
            yield event

    async def stream_turn(
        self,
        session_id: str,
        message: str,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer=None,
        on_llm_payload=None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the events of one turn (``run_start`` .. ``run_end``)."""
        state, run_kwargs = await self._load_or_seed(session_id, message, owner=owner)
        async for event in self.graph.stream(
            state,
            tools=self.tools,
            reducers=self.reducers,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations or self.max_iterations,
            tracer=tracer,
            on_llm_payload=on_llm_payload,
            **run_kwargs,
        ):
            yield event

    async def last_reply(self, session_id: str, *, owner: str = DEFAULT_OWNER) -> str:
        """Return the latest assistant reply for *session_id* (``""`` if none).

        Reads the durable checkpoint, so it works even for agents that do
        not stream tokens (e.g. tool-using agents): the CLI prints this at
        the end of a turn instead of relying on ``token`` events alone.
        """
        saved = await self.checkpointer.load(session_id, owner=owner)
        if saved is None:
            return ""
        for message in reversed(saved.state.get(self.messages_key) or []):
            if message.get("role") == "assistant":
                return str(message.get("content", ""))
        return ""

    async def pending(
        self, session_id: str, *, owner: str = DEFAULT_OWNER
    ) -> dict | None:
        """Return the interrupt this session is paused on, or ``None``.

        The interrupt bookkeeping lives in durable state: when ``graph.run``
        pauses on an :class:`~draf.node.Interrupt` it writes a ``__interrupt__``
        entry into the saved checkpoint.  This reads it back so the caller —
        without a try/except or an in-memory ``pending`` map — can tell whether
        the next message is a fresh turn or the operator's answer to resume.
        """
        saved = await self.checkpointer.load(session_id, owner=owner)
        if saved is None:
            return None
        return saved.state.get(_INTERRUPT_KEY)

    async def turn(
        self,
        session_id: str,
        message: str,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer=None,
        on_llm_payload=None,
    ) -> TurnResult:
        """Run one turn, resuming a paused session transparently.

        This is the single entry point apps use instead of calling
        ``run_turn``/``resume_turn`` and catching
        :class:`~draf.node.GraphInterrupt` themselves::

            result = await assistant.turn(session_id, message)
            if result.waiting:
                # surface result.prompt to the operator, await their answer
                ...                       # and call turn() again with it
            else:
                print(result.reply)

        * If the session is paused on an interrupt (:meth:`pending`), *message*
          is the operator's answer and the run resumes from the checkpoint.
        * Otherwise *message* starts (or continues) the conversation.
        * A pause is **not** raised to the caller: it is folded into the
          returned :class:`TurnResult` (``waiting=True`` with the prompt and
          key), so the loop above keeps working across an arbitrary number of
          interrupts (e.g. a "rework" branch that re-asks).
        """
        from draf.node import GraphInterrupt

        try:
            resume = await self.pending(session_id, owner=owner)
            if resume is not None:
                state = await self.resume_turn(
                    session_id,
                    {resume["key"]: message},
                    owner=owner,
                    max_iterations=max_iterations or self.max_iterations,
                    tracer=tracer,
                    on_llm_payload=on_llm_payload,
                )
            else:
                state = await self.run_turn(
                    session_id,
                    message,
                    owner=owner,
                    max_iterations=max_iterations or self.max_iterations,
                    tracer=tracer,
                    on_llm_payload=on_llm_payload,
                )
            return TurnResult(
                session_id=session_id,
                reply=await self.last_reply(session_id, owner=owner),
                state=state,
            )
        except GraphInterrupt as exc:
            return TurnResult(
                session_id=session_id,
                reply=await self.last_reply(session_id, owner=owner),
                waiting=True,
                prompt=exc.prompt,
                key=exc.key,
            )

    async def stream(
        self,
        session_id: str,
        message: str,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer=None,
        on_llm_payload=None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one turn, resuming a paused session transparently.

        The streaming equivalent of :meth:`turn`.  Relays the underlying
        ``graph.stream`` events verbatim but picks the resume source from the
        durable interrupt state (:meth:`pending`) instead of a caller-provided
        ``pending`` map.  A re-work pause surfaces an ``interrupt`` event (with
        ``key``/``prompt`` in its ``data``) and the stream ends there — call
        this again with the operator's answer to continue.
        """
        resume = await self.pending(session_id, owner=owner)
        source = (
            self.stream_resume(
                session_id,
                {resume["key"]: message},
                owner=owner,
                max_iterations=max_iterations or self.max_iterations,
                tracer=tracer,
                on_llm_payload=on_llm_payload,
            )
            if resume is not None
            else self.stream_turn(
                session_id,
                message,
                owner=owner,
                max_iterations=max_iterations or self.max_iterations,
                tracer=tracer,
                on_llm_payload=on_llm_payload,
            )
        )
        async for event in source:
            yield event
