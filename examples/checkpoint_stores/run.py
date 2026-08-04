"""Run any durable workflow from a YAML file — CLI emulation.

Each store example has its own workflow.yaml with a ``checkpoint:``
block describing the backend and a ``plugins:`` entry loading the
``failing`` node (see ``failing.py``), which raises once on the first
run (simulating a transient crash).  A re-run with the same
``checkpoint_id`` resumes from the saved checkpoint and completes.

Usage:
    uv run python examples/checkpoint_stores/run.py <path/to/workflow.yaml>
"""

import asyncio
import os
import sys

from draf.yaml import load_workflow


def _resolve_checkpoint_config(config: dict, base_dir: str) -> dict:
    """Resolve relative paths in a ``checkpoint:`` block against base_dir."""
    result = dict(config)
    for key in ("path",):
        if key in result and not os.path.isabs(result[key]):
            result[key] = os.path.join(base_dir, result[key])
    return result


def _make_checkpointer(config: dict):
    ctype = config.get("type")
    if ctype == "file":
        from draf.checkpoint import JSONFileCheckpointer

        return JSONFileCheckpointer(config.get("path", "checkpoints"))
    if ctype == "sqlite":
        from draf.checkpoint import SQLiteCheckpointer

        return SQLiteCheckpointer(config.get("path", "checkpoints.db"))
    if ctype == "pg":
        from draf.checkpoint.pg import PGCheckpointer

        return PGCheckpointer(config.get("dsn"), config.get("table", "checkpoints"))
    raise ValueError(f"unknown checkpoint type: {ctype}")


async def main(path: str):
    graph, tools, state, reducers = load_workflow(path)

    base_dir = os.path.dirname(os.path.abspath(path))
    cp_config = _resolve_checkpoint_config(_read_checkpoint_block(path), base_dir)
    checkpoint_id = cp_config.pop("checkpoint_id", "demo-run")
    checkpointer = _make_checkpointer(cp_config)

    # fresh start so the crash/resume dance is reproducible
    await checkpointer.delete(checkpoint_id)

    for attempt in (1, 2):
        try:
            result = await graph.run(
                state,
                tools=tools,
                reducers=reducers,
                checkpointer=checkpointer,
                checkpoint_id=checkpoint_id,
            )
            print(f"Run {attempt}: success -> {result}")
            break
        except RuntimeError as e:
            print(f"Run {attempt}: crashed ({e}), checkpoint saved")


def _read_checkpoint_block(path: str) -> dict:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("checkpoint", {})


if __name__ == "__main__":
    default = "file/workflow.yaml"
    path = sys.argv[1] if len(sys.argv) > 1 else default
    asyncio.run(main(path))
