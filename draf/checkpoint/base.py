"""Checkpointing primitives for durable graph execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

#: Default owner when none is given.  Keeps single-tenant callers working
#: while every checkpoint still lives under a named namespace, so data can
#: later be split out per user without a migration of checkpoint IDs.
DEFAULT_OWNER = "default"


@dataclass
class Checkpoint:
    """A snapshot of execution that can be resumed from.

    Attributes:
        state: Workflow state data (JSON-serializable dict).
        next_node_id: ID of the node to execute on resume.
            ``None`` means the graph completed.
        iteration: Number of completed node executions so far.
    """

    state: dict
    next_node_id: str | None
    iteration: int


class Checkpointer(Protocol):
    """Interface for persisting execution checkpoints.

    Implementations must be safe to call concurrently for different
    checkpoint IDs and must persist *atomically* enough that a crash
    never leaves a partially-written checkpoint.

    The optional *owner* scopes a checkpoint to a user/session/tenant.
    Two checkpoints with the same ID but different owners never collide.
    When omitted the default owner (:data:`DEFAULT_OWNER`, ``"default"``)
    is used, keeping single-tenant callers unchanged.

    Use a distinct owner per end-user (e.g. a user id or session id) so
    every tenant's runs are isolated from the others and can be listed
    with :meth:`list`.
    """

    async def save(
        self, checkpoint_id: str, checkpoint: Checkpoint, *, owner: str = DEFAULT_OWNER
    ) -> None:
        """Persist *checkpoint* under *checkpoint_id* for *owner*."""
        ...

    async def load(
        self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER
    ) -> Checkpoint | None:
        """Return the saved checkpoint for *owner*, or ``None`` if never saved."""
        ...

    async def delete(self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER) -> None:
        """Remove a saved checkpoint. No-op if it does not exist."""
        ...

    async def list(self, owner: str = DEFAULT_OWNER) -> list[str]:
        """Return all checkpoint IDs persisted for *owner*."""
        ...

    async def cleanup(
        self,
        *,
        owner: str | None = None,
        max_age: float | None = None,
        keep_last: int | None = None,
    ) -> int:
        """Delete stale checkpoints; returns how many were removed.

        ``owner=None`` cleans up every owner; otherwise only that owner.
        ``max_age`` removes checkpoints last written more than that many
        seconds ago.  ``keep_last`` retains the *N* most recently written
        checkpoints per owner (after any ``max_age`` pruning) and deletes
        the rest.  When both are omitted nothing is deleted.
        """
        ...


def checkpoint_to_dict(cp: Checkpoint) -> dict[str, Any]:
    """Convert a checkpoint to a JSON-serializable dict."""
    return {
        "state": cp.state,
        "next_node_id": cp.next_node_id,
        "iteration": cp.iteration,
    }


def checkpoint_from_dict(data: dict[str, Any]) -> Checkpoint:
    """Reconstruct a checkpoint from a dict produced by :func:`checkpoint_to_dict`."""
    return Checkpoint(
        state=data.get("state", {}),
        next_node_id=data.get("next_node_id"),
        iteration=data.get("iteration", 0),
    )
