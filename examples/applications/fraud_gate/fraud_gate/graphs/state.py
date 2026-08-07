"""Typed graph state for the fraud-gate example.

A payment enters as a JSON message, is analysed by an LLM
(:mod:`fraud_gate.graphs.prompts`) and routed with :class:`teff.node.Command`.
The transaction fields are stored as **flat, top-level state keys** because
node prompts render ``{key}`` placeholders directly from state (no nested
dereferencing).

``events`` accumulates the event log (append reducer) — which is exactly the
kind of state a ``Command`` updates in the same turn it routes.  The
per-request scratch (the transaction fields, ``analysis``, ``risk``,
``decision``, ``reason``, ``review_decision``, ``final``) is transient and
reset by the entry node so one request never leaks into the next.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from teff.state import reducers_from_typeddict


def add_messages(current: list | None, new: list | None) -> list:
    """Reducer: append entries to an accumulator (used for the event log)."""
    return list(current or []) + list(new or [])


class Transaction(TypedDict, total=False):
    """The payment being screened (fields are optional).

    ``note`` is free text the analyser LLM reads — the actual "work".
    """

    id: str
    amount: float
    merchant: str
    country: str
    ip_country: str
    note: str


class RiskVerdict(TypedDict):
    """Structured output of the analyser LLM (``output_type``)."""

    risk: float
    signals: str


class FinalVerdict(TypedDict):
    """Structured output of the finalizer LLM (``output_type``)."""

    decision: str
    summary: str


class FraudState(TypedDict):
    """State carried through the fraud-gate graph."""

    #: Event log (append reducer).
    events: Annotated[list, add_messages]

    #: The payment under evaluation (flat top-level fields for the prompts).
    tx_id: str
    tx_amount: float
    tx_merchant: str
    tx_country: str
    tx_ip_country: str
    tx_note: str

    #: The analyser LLM's :class:`RiskVerdict`.
    analysis: dict

    #: Risk score in [0, 1] (copy of ``analysis.risk`` for the API).
    risk: float

    #: ``approve`` | ``review`` | ``deny``.
    decision: str

    #: Human-readable risk breakdown (from ``analysis.signals``).
    reason: str

    #: Human answer on the mid-risk review gateway (``pass`` / ``deny``).
    review_decision: str

    #: Finalizer's :class:`FinalVerdict` (or the inline deny verdict).
    final: dict


STATE_REDUCERS: dict[str, Any] = reducers_from_typeddict(FraudState)


def initial_state() -> FraudState:
    """Fresh, empty state for a new review request."""
    return {
        "events": [],
        "tx_id": "",
        "tx_amount": 0.0,
        "tx_merchant": "",
        "tx_country": "",
        "tx_ip_country": "",
        "tx_note": "",
        "analysis": {},
        "risk": 0.0,
        "decision": "",
        "reason": "",
        "review_decision": "",
        "final": {},
    }
