"""High-level graph execution helper."""

from draf.graph import Graph


async def execute_graph(
    graph: Graph,
    state: dict,
    tools: list,
    registry,
) -> dict:
    """Execute a graph with the given state, tools, and registry.

    Convenience wrapper around ``graph.run()``.
    """
    return await graph.run(state=state, tools=tools, registry=registry)
