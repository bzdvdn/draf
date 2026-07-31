"""Durable graph on any checkpoint backend, built with the Python Flow API.

Mirrors the YAML workflows next to it: same graph, same crash/resume
dance, but assembled from code instead of loaded from workflow.yaml.

Usage:
    uv run python examples/checkpoint_stores/flow.py [backend]

Supported backends: file, sqlite, pg (default: file).
See each backend's README for install steps.
"""

import asyncio
import os
import sys

from draf.flow import Flow
from draf.node import Node, Transform

_HERE = os.path.dirname(os.path.abspath(__file__))

BACKEND_CONFIGS = {
    "file": {
        "factory": "json_file",
        "path": os.path.join(_HERE, "file", "checkpoints"),
    },
    "sqlite": {
        "factory": "sqlite",
        "path": os.path.join(_HERE, "sqlite", "checkpoints.db"),
    },
    "pg": {
        "factory": "pg",
        "dsn": "postgresql://postgres:postgres@localhost:5433/postgres",
    },
}

# Simulates a transient external failure.  Lives outside the workflow
# state because state is restored to the pre-node checkpoint on resume.
_crash_once = {"armed": True}


class FailingNode(Node):
    """Raises on the first execution, succeeds afterwards."""

    type = "failing"

    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__(**(config or {}), **kwargs)

    async def execute(self, ctx, state):
        if _crash_once["armed"]:
            _crash_once["armed"] = False
            raise RuntimeError("simulated transient failure")
        state["recovered"] = True
        return state


def _make_checkpointer(config: dict):
    factory = config.get("factory")
    if factory == "json_file":
        from draf.checkpoint import JSONFileCheckpointer

        return JSONFileCheckpointer(config.get("path"))
    if factory == "sqlite":
        from draf.checkpoint import SQLiteCheckpointer

        return SQLiteCheckpointer(config.get("path"))
    if factory == "pg":
        from draf.checkpoint.pg import PGCheckpointer

        return PGCheckpointer(config.get("dsn"))
    raise ValueError(f"unknown checkpoint factory: {factory}")


async def main(backend: str):
    if backend not in BACKEND_CONFIGS:
        print(f"unknown backend: {backend}")
        print("choose one of:", ", ".join(BACKEND_CONFIGS))
        raise SystemExit(1)

    flow = Flow(f"durable_{backend}")
    flow.step(
        Transform({"action": "uppercase", "input_key": "text", "output_key": "shout"})
    )
    flow.step(FailingNode({}))
    flow.step(
        Transform({"action": "value", "value": "finished", "output_key": "status"})
    )
    graph = flow.compile()

    checkpointer = _make_checkpointer(BACKEND_CONFIGS[backend])

    print("Backend:", backend)
    await checkpointer.delete("demo-run")  # fresh start for reproducibility

    for attempt in (1, 2):
        try:
            result = await graph.run(
                state={"text": "durable"},
                checkpointer=checkpointer,
                checkpoint_id="demo-run",
            )
            print(f"Run {attempt}: success -> {result}")
            break
        except RuntimeError as e:
            print(f"Run {attempt}: crashed ({e}), checkpoint saved")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "file"))
