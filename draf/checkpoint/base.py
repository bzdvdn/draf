"""Checkpointing primitives for durable graph execution."""

from dataclasses import dataclass
from typing import Any, Protocol


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
    """

    async def save(self, checkpoint_id: str, checkpoint: Checkpoint) -> None:
        """Persist *checkpoint* under *checkpoint_id*."""
        ...

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Return the saved checkpoint, or ``None`` if never saved."""
        ...

    async def delete(self, checkpoint_id: str) -> None:
        """Remove a saved checkpoint. No-op if it does not exist."""
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
