"""Transient state keys.

These keys are recomputed every run and are not part of the durable
conversation history, so they are cleared at the start of each fresh turn.
"""

from __future__ import annotations

#: Transient state keys recomputed each run (not part of the conversation).
TRANSIENT_KEYS = (
    "input",
    "output",
    "_coordinator_messages",
    "plan",
    "estimate",
    "material_findings",
    "qa_feedback",
)
