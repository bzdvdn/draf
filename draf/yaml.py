"""YAML serialisation and deserialisation for graphs."""

import os

import yaml

from draf.graph import Graph, Edge


def from_yaml(source: str) -> Graph:
    """Parse a YAML string or file path into a ``Graph``.

    The YAML format::

        name: my-graph
        steps:
          - id: start
            type: transform
            config: {action: "uppercase"}
        edges:
          - from: start
            to: next

    If *source* is an existing file path it is read from disk;
    otherwise it is treated as a raw YAML string.

    Args:
        source: YAML string or path to a ``.yaml`` file.

    Returns:
        A compiled ``Graph`` ready for execution.
    """
    if os.path.exists(source):
        with open(source) as f:
            data = yaml.safe_load(f)
    else:
        data = yaml.safe_load(source)

    from draf.node.registry import default_registry

    nodes = {}
    edges = []
    entry_point = None

    for step in data.get("steps", []):
        sid = step["id"]
        stype = step["type"]
        config = step.get("config", {})
        node = default_registry.create(stype, config)
        node.type = sid
        nodes[sid] = node
        if entry_point is None:
            entry_point = sid

    for edge_data in data.get("edges", []):
        edges.append(Edge(
            source_id=edge_data["from"],
            target_id=edge_data["to"],
            condition=edge_data.get("condition"),
        ))

    return Graph(nodes=nodes, edges=edges, entry_point=entry_point or "")


def graph_to_yaml(graph: Graph) -> str:
    """Serialize a ``Graph`` instance to a YAML string."""
    steps = []
    for nid, node in graph.nodes.items():
        steps.append({
            "id": nid,
            "type": getattr(node, "type", nid),
            "config": getattr(node, "config", {}),
        })

    edges = []
    for e in graph.edges:
        entry = {"from": e.source_id, "to": e.target_id}
        if e.condition:
            entry["condition"] = e.condition
        edges.append(entry)

    data = {
        "name": "graph",
        "steps": steps,
        "edges": edges,
    }
    return yaml.dump(data, default_flow_style=False)
