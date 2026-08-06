"""Graph serialization: Mermaid diagrams and YAML topology."""

from __future__ import annotations

from draf.graph.edge import _ERROR_CONDITION


def _mmq(value: str) -> str:
    """Quote an identifier for Mermaid so special characters are safe."""
    return f'"{value.replace(chr(34), "")}"'


def _mme(value: str) -> str:
    """Escape a label for Mermaid (quotes and backslashes)."""
    return value.replace("\\", "\\\\").replace(chr(34), "&#34;")


def _condition_label(condition) -> str:
    """Render an edge condition as a label string (callables by name)."""
    if callable(condition):
        name = getattr(condition, "__name__", None)
        return f"when:{name}" if name else "when:<callable>"
    return str(condition)


def to_mermaid(graph, show_conditions: bool = True) -> str:
    """Render *graph* as a Mermaid flowchart diagram.

    Produces a ``flowchart TD`` definition: every node becomes a box
    labelled ``node_id[node.type]`` and every edge an arrow.  The entry
    point is filled blue, ``__error__`` edges are dashed and red, and
    conditional edges carry their condition as an edge label (when
    *show_conditions* is true).

    Args:
        graph: The graph to render (exposes ``nodes``, ``edges``,
            ``entry_point``).
        show_conditions: Annotate conditional edges with their condition.

    Returns:
        The Mermaid diagram as a string (no code fence).
    """
    lines = ["flowchart TD"]
    for node_id, node in graph.nodes.items():
        label = f"{node_id}[{node.type}]"
        lines.append(f'    {_mmq(node_id)}["{_mme(label)}"]')
    lines.append(f"    class {_mmq(graph.entry_point)} entry;")
    for edge in graph.edges:
        src = _mmq(edge.source_id)
        dst = _mmq(edge.target_id)
        if edge.condition == _ERROR_CONDITION:
            lines.append(f"    {src} -.->|error| {dst}")
        elif edge.condition and show_conditions:
            label = _condition_label(edge.condition)
            lines.append(f'    {src} -->|"{_mme(label)}"| {dst}')
        else:
            lines.append(f"    {src} --> {dst}")
    lines.append("    classDef entry fill:#bde0fe;")
    lines.append("    classDef error stroke:#ff5252,stroke-width:2px;")
    return "\n".join(lines)


__all__ = ["to_mermaid", "_mmq", "_mme"]
