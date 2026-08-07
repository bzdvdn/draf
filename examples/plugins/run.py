"""Run the plugins example workflow (offline, no LLM, no API keys).

Loads ``workflow.yaml`` — the plugin discovery happens inside
``load_workflow``, which imports ``plugins/nodes.py`` and ``plugins/tools.py``
(and any ``plugins/`` folder) before validation.  The graph runs
custom node types that call the custom tool types directly via
``ctx.tools``.

Usage:
    python examples/plugins/run.py
"""

import asyncio
import json
from pathlib import Path

from teff.yaml import load_workflow

HERE = Path(__file__).resolve().parent


async def main() -> None:
    graph, tools, initial, reducers = load_workflow(str(HERE / "workflow.yaml"))

    result = await graph.run(state=initial, tools=tools, reducers=reducers)

    print("slug:  ", result["slug"])
    print("count: ", result["count"])
    print("report:")
    print(result["report"])
    assert json.loads(result["report"]) == {
        "slug": result["slug"],
        "count": result["count"],
    }


if __name__ == "__main__":
    asyncio.run(main())
