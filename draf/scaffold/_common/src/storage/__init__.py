"""Session persistence — durable checkpoints for conversations.

A checkpoint is saved after every node, so a crash never loses work and a
conversation can continue across process restarts by re-using its session
id.  :func:`build_checkpointer` picks the backend from config:

* ``database_url`` / ``checkpoint_db`` set to a Postgres DSN  ->
  :class:`~draf.checkpoint.PGCheckpointer` (shared, multi-process safe).
* ``checkpoint_db`` set to a path  -> :class:`~draf.checkpoint.SQLiteCheckpointer`.
* otherwise  -> per-session JSON files in ``data/checkpoints/``.

HOW TO EXTEND
    Swap the backend once (here), not per call site.
"""

from __future__ import annotations

from pathlib import Path

from draf.checkpoint import (
    DEFAULT_OWNER,
    Checkpoint,
    Checkpointer,
    JSONFileCheckpointer,
    PGCheckpointer,
    SQLiteCheckpointer,
)
from src.graphs.state import initial_state

#: Relative to the project root; points at ``data/checkpoints/``.
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints"

#: Transient state keys recomputed each run (not part of the conversation).
_TRANSIENT_KEYS = (
    "next_agent",
    "input",
    "plan",
    "draft",
    "review",
    "supervisor_rounds",
)


def _is_dsn(value: str | Path | None) -> bool:
    """True when *value* is a Postgres connection string (vs a file path)."""
    return str(value or "").startswith(("postgres://", "postgresql://"))


def build_checkpointer(
    directory: str | Path | None = None,
    *,
    checkpoint_db: str | Path | None = None,
) -> Checkpointer:
    """Build the session checkpointer used by ``graph.run(..., checkpointer=...)``.

    *checkpoint_db* selects the durable backend: a Postgres DSN
    (``postgres://...``) uses :class:`PGCheckpointer`, a path uses
    :class:`SQLiteCheckpointer`, and ``None`` keeps the per-session JSON
    files.
    """
    if _is_dsn(checkpoint_db):
        return PGCheckpointer(str(checkpoint_db))
    if checkpoint_db is not None:
        return SQLiteCheckpointer(str(checkpoint_db))
    return JSONFileCheckpointer(str(directory or DEFAULT_CHECKPOINT_DIR))


async def load_or_seed(
    graph,
    checkpointer,
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
    saved = await checkpointer.load(session_id, owner=owner)
    if saved is None:
        state: dict = dict(initial_state())
        state["messages"] = [{"role": "user", "content": message}]
        return state, {}

    state = dict(saved.state)
    messages = list(state.get("messages") or [])
    messages.append({"role": "user", "content": message})
    state["messages"] = messages
    for key in _TRANSIENT_KEYS:
        state[key] = ""
    await checkpointer.save(
        session_id,
        Checkpoint(state=state, next_node_id=graph.entry_point, iteration=0),
        owner=owner,
    )
    return {}, {}
