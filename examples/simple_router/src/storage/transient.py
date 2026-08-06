"""Transient state keys.

These keys are recomputed every run and are not part of the durable
conversation state, so they are excluded from checkpoints.
"""

from __future__ import annotations

#: Transient state keys recomputed each run (not part of the conversation).
TRANSIENT_KEYS = ("next_agent", "input", "code", "talk", "supervisor_rounds")
