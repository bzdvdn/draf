"""Transient state keys.

These keys are recomputed every run and are not part of the durable
conversation state, so they are excluded from checkpoints.
"""

from __future__ import annotations

#: Transient state keys recomputed each run (not part of the conversation).
TRANSIENT_KEYS = (
    "next_agent",
    "input",
    "direct_reply",
    "plan",
    "plan_approved",
    "plan_ok",
    "plan_verdict",
    "plan_rounds",
    "estimate",
    "material_findings",
    "qa_feedback",
    "qa_verdict",
    "qa_ok",
    "qa_rounds",
    "estim_approved",
    "est_ok",
    "est_verdict",
    "est_rounds",
    "final_answer",
    "supervisor_rounds",
)
