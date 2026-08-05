"""Graph topology snapshot — the node/edge shape for the dashboard."""

from __future__ import annotations

from typing import Any

from draf.observability.model import GraphTopology


def topology_from_graph(graph) -> GraphTopology:
    """Capture ``{nodes, edges}`` from a compiled :class:`~draf.graph.Graph`.

    Each node is ``{"id", "type"}``; each edge is ``{"source", "target",
    "condition"}`` (``condition`` omitted when unconditional).
    """
    nodes = [
        {"id": node_id, "type": node.type} for node_id, node in graph.nodes.items()
    ]
    edges: list[dict[str, Any]] = []
    for edge in graph.edges:
        item: dict[str, Any] = {"source": edge.source_id, "target": edge.target_id}
        if getattr(edge, "condition", None):
            item["condition"] = edge.condition
        edges.append(item)
    return GraphTopology(nodes=nodes, edges=edges)
