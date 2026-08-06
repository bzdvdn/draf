"""Fraud-gate graph — a production use case for :class:`draf.node.Command`.

Two LLM nodes do the actual analytic work — an analyser reads the payment's
free-text ``note`` and returns a structured risk score; a finalizer writes
the human-facing outcome.  The routing between them is driven by a
:class:`draf.node.Command` returned from the ``router`` node, because the
next step depends on the **value of the score** the model just produced and
cannot be statically wired:

    ingest -> analyser (LLM) -> router --(approve)---> finalizer (goto)
                                     --(review)---> review_gate -> finalizer
                                     --(deny)-----> END (Command.STOP)

Three behaviours:

* **Approve** — ``Command(update=..., goto="finalize")`` jumps straight to
  the finalizer even though there is **no** ``router -> finalize`` edge.
* **Mid-risk review** — ``Command(update=...)`` (no ``goto``), so the normal
  ``router -> review_gate`` edge runs; a human approves/declines via an
  :class:`~draf.node.Interrupt` and the run pauses for a durable resume.
* **Deny** — ``Command(update=..., goto=Command.STOP)`` terminates
  immediately, blocking the payment without running any further node.

Every ``Command`` also **appends to the event log** (``messages`` reducer),
so routing and state-writing happen in the same return value.
"""

from __future__ import annotations

from draf.flow import Flow
from draf.provider import ProviderRegistry

from .nodes import Ingest, Router
from .prompts import (
    ANALYSER_PROMPT,
    ANALYSER_SYSTEM,
    FINALIZER_PROMPT,
    FINALIZER_SYSTEM,
)
from .state import FinalVerdict, RiskVerdict

MODEL_DEFAULT = "llama3.1:8b"

#


def build_flow(
    model: str = MODEL_DEFAULT,
    *,
    provider: str = "ollama",
) -> Flow:
    """Assemble the fraud-gate flow; returns the compiled graph.

    *provider*/*model* are threaded into every LLM node (per-node) so the
    graph never touches framework-global defaults.
    """
    flow = (
        Flow(
            "fraud_gate",
            providers=ProviderRegistry.from_presets(provider),
            default_provider=provider,
        )
        .step(Ingest(), id="ingest")
        .llm(
            model=model,
            provider=provider,
            system=ANALYSER_SYSTEM,
            prompt=ANALYSER_PROMPT,
            output_key="analysis",
            output_type=RiskVerdict,
            id="analyser",
        )
        .step(Router(), id="router")
        .interrupt(
            key="review_decision",
            prompt=(
                "Скрипт требует проверки (риск {risk}, сигналы: {reason}). "
                "Ответь: pass или deny."
            ),
            id="review_gate",
        )
        .llm(
            model=model,
            provider=provider,
            system=FINALIZER_SYSTEM,
            prompt=FINALIZER_PROMPT,
            output_key="final",
            output_type=FinalVerdict,
            id="finalize",
        )
    )

    return flow
