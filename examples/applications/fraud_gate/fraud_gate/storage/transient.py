"""Transient state keys.

These are recomputed for every request and are not part of the durable
history, so they are cleared at the start of each fresh review (the entry
``Ingest`` node resets them regardless).
"""

from __future__ import annotations

#: Transient state keys recomputed each run.
TRANSIENT_KEYS = (
    "tx",
    "risk",
    "decision",
    "reason",
    "review_decision",
    "final",
)
