"""Transient state keys.

These keys are recomputed every run and are not part of the durable
conversation history, so they are cleared at the start of each fresh turn
(the entry ``ContextBuilder`` resets them regardless).
"""

from __future__ import annotations

#: Transient state keys recomputed each run (not part of the conversation).
TRANSIENT_KEYS = (
    "next_agent",
    "supervisor_rounds",
    "billing",
    "incident",
    "deploy",
    "fallback",
    "deploy_approved",
    "input",
    "final",
)
