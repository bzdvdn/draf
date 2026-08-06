"""Review service — the domain orchestration behind the API gate.

Owns the business flow: generate a durable session, drive one review through
the :class:`~draf.assistant.Assistant`, read the durable session state and
normalise it into pure :mod:`~fraud_gate.domain.models` value objects.  The
HTTP router stays thin — it only marshals pydantic request/response schemas
and delegates here.

The ``Assistant`` owns the pause/resume lifecycle: its ``run`` detects a
paused interrupt from the durable checkpoint and resumes with the analyst's
answer, so this service never sees a ``GraphInterrupt``.  On a pause the
returned ``TurnResult`` carries no state, so the decision is read back from
the durable checkpoint (``assistant.get_state``) and a stray empty ``final``
is folded to ``None``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

from draf import Assistant
from fraud_gate.domain.models import Conflict, DecideOutcome, ReviewOutcome


def _session_id(provided: str | None) -> str:
    return provided or uuid.uuid4().hex


def _state_get(state: dict | None, key: str):
    return (state or {}).get(key)


def _final_or_none(state: dict | None) -> dict | None:
    """Return ``state['final']`` only when it is a structured verdict dict."""
    if not state:
        return None
    final = state.get("final")
    return final if isinstance(final, dict) else None


class ReviewService:
    """Screen payments and resume paused reviews via the domain graph."""

    def __init__(self, assistant: Assistant):
        self.assistant = assistant

    async def screen(
        self,
        owner: str,
        tx: dict,
        session_id: str | None = None,
        *,
        tracer: Any = None,
        on_llm_payload: Callable | None = None,
        trace_finish: Callable[[], str | None] | None = None,
    ) -> ReviewOutcome:
        """Screen one payment; may pause for a human analyst.

        The transaction is handed to the graph as a JSON message and the
        analyser LLM turns its free ``note`` into a structured risk score.

        ``tracer`` / ``on_llm_payload`` are the framework's observability
        hooks forwarded to the run; ``trace_finish`` persists the captured
        run and returns its id (set onto the outcome's ``run_id``).
        """
        sid = _session_id(session_id)
        result = await self.assistant.run(
            sid,
            json.dumps(tx),
            owner=owner,
            max_iterations=20,
            tracer=tracer,
            on_llm_payload=on_llm_payload,
        )
        state = (
            result.state
            if result.state is not None
            else await self.assistant.get_state(sid, owner=owner)
        )
        return ReviewOutcome(
            session_id=sid,
            decision=_state_get(state, "decision"),
            risk=_state_get(state, "risk"),
            waiting=result.waiting,
            prompt=result.prompt if result.waiting else None,
            final=_final_or_none(state),
            events=list((state or {}).get("events") or []),
            run_id=trace_finish() if trace_finish else None,
        )

    async def decide(
        self,
        owner: str,
        session_id: str,
        answer: str,
        *,
        tracer: Any = None,
        on_llm_payload: Callable | None = None,
        trace_finish: Callable[[], str | None] | None = None,
    ) -> DecideOutcome:
        """Resume a paused mid-risk review with the analyst's answer.

        Only a session genuinely paused on the review gateway can be resumed;
        otherwise the analyst's answer would be treated as a **fresh** user
        message and screen an empty transaction.
        """
        if await self.assistant.pending(session_id, owner=owner) is None:
            raise Conflict(session_id)

        result = await self.assistant.run(
            session_id,
            answer,
            owner=owner,
            max_iterations=20,
            tracer=tracer,
            on_llm_payload=on_llm_payload,
        )
        state = (
            result.state
            if result.state is not None
            else await self.assistant.get_state(session_id, owner=owner)
        )
        return DecideOutcome(
            session_id=session_id,
            decision=_state_get(state, "decision"),
            waiting=result.waiting,
            final=_final_or_none(state),
            run_id=trace_finish() if trace_finish else None,
        )

    async def get(self, owner: str, session_id: str) -> dict | None:
        """Return the durable state of a review (or ``None`` if it does not exist)."""
        return await self.assistant.get_state(session_id, owner=owner)
