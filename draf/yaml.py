"""YAML serialisation and deserialisation for graphs."""

import os
import re

import yaml

from draf.graph import Graph, Edge
from draf.errors import ConfigError
from draf.node.node import Node
from draf.tool.tool import Tool
from draf.tool.registry import default_tool_registry
from draf.state.state import Reducer, reducers_from_yaml_schema
from draf.yaml_schema import validate_workflow, raise_for_validation


def _safe_load(source):
    """Load YAML, surfacing parse errors as ``ConfigError``."""
    try:
        return yaml.safe_load(source)
    except (yaml.YAMLError, ValueError) as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc


_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env(value):
    """Replace ``${ENV_VAR}`` references in a YAML structure with values.

    Recursively walks strings inside mappings and lists.  A reference to a
    missing variable is left unchanged (rather than raising) so templates
    stay valid offline; set the variable to inject the secret.
    """
    if isinstance(value, str):
        return _ENV_VAR.sub(
            lambda m: os.environ.get(m.group(1), m.group(0)), value
        )
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


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
            data = _safe_load(f)
    else:
        data = _safe_load(source)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("workflow must be a mapping")
    data = _interpolate_env(data)
    label = source if os.path.exists(source) else "workflow"
    raise_for_validation(validate_workflow(data), source=label)

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
        edges.append(
            Edge(
                source_id=edge_data["from"],
                target_id=edge_data["to"],
                condition=edge_data.get("condition"),
            )
        )

    return Graph(nodes=nodes, edges=edges, entry_point=entry_point or "")


def workflow_to_yaml(
    graph: Graph,
    *,
    tools: list[Tool] | None = None,
    initial: dict | None = None,
    reducers: dict[str, Reducer] | None = None,
    name: str = "graph",
) -> str:
    """Serialize a ``Graph`` (plus optional tools/state) to a workflow YAML.

    ``steps`` and ``edges`` come from the graph; ``tools`` are written
    from their ``name`` and (when present) their ``config`` attribute;
    ``reducers`` become the ``state.schema`` block (string reducers only)
    and *initial* becomes ``state.initial``.

    The output validates with :func:`validate_workflow` and round-trips
    through :func:`load_workflow`.
    """
    steps = []
    for nid, node in graph.nodes.items():
        steps.append(
            {
                "id": nid,
                "type": getattr(type(node), "type", None) or getattr(node, "type", nid),
                "config": getattr(node, "config", {}) or {},
            }
        )

    edges = []
    for e in graph.edges:
        entry = {"from": e.source_id, "to": e.target_id}
        if e.condition:
            entry["condition"] = e.condition
        edges.append(entry)

    data: dict = {"name": name, "steps": steps, "edges": edges}
    if tools:
        data["tools"] = [{"type": t.name, "config": _tool_config(t)} for t in tools]
    if reducers or initial:
        from draf.state.state import reducers_to_yaml_schema

        state: dict = {}
        if reducers:
            state["schema"] = reducers_to_yaml_schema(reducers)
        if initial:
            state["initial"] = initial
        data["state"] = state
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def _tool_config(tool: Tool) -> dict:
    cfg = getattr(tool, "config", None)
    if isinstance(cfg, dict):
        return cfg
    return {}


def graph_to_yaml(graph: Graph) -> str:
    """Serialize a ``Graph`` instance to a YAML string.

    This is shorthand for :func:`workflow_to_yaml` without tools or state.
    """
    return workflow_to_yaml(graph)


def load_workflow(path: str) -> tuple[Graph, list[Tool], dict, dict[str, Reducer]]:
    """Load a complete workflow from a YAML file (graph + tools + state).

    YAML format::

        name: my-workflow
        tools:
          - type: calculator
          - type: shell
            config: {root_dir: /tmp}
        state:
          schema:
            messages:
              reducer: append
              type: list
          initial:
            status: active
        steps:
          - id: step1
            type: transform
            config: {action: uppercase, input_key: text, output_key: out}
        edges:
          - from: step1
            to: step2

    Returns:
        A ``(Graph, tools_list, initial_state, reducers)`` tuple ready for ``graph.run()``.
    """
    with open(path) as f:
        data = _safe_load(f)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("workflow must be a mapping")

    # Resolve ${ENV} references across the whole document (tools, steps,
    # state), then load plugins so custom node/tool types validate below.
    data = _interpolate_env(data)
    from draf.plugins import load_plugins_from_document

    load_plugins_from_document(data, os.path.dirname(os.path.abspath(path)))

    import draf.tool.builtin  # noqa: F401 — registers built-in tools
    import draf.rag  # noqa: F401 — registers the "rag" tool

    raise_for_validation(validate_workflow(data), source=path)

    from draf.node.registry import default_registry

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    entry_point: str | None = None

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
        edges.append(
            Edge(
                source_id=edge_data["from"],
                target_id=edge_data["to"],
                condition=edge_data.get("condition"),
            )
        )

    tools: list[Tool] = []
    base_dir = os.path.dirname(os.path.abspath(path))
    for td in data.get("tools", []):
        ttype = td["type"]
        tconfig = td.get("config", {})
        if ttype == "rag":
            tconfig = _resolve_rag_config(tconfig, base_dir)
        tools.append(default_tool_registry.create(ttype, tconfig))

    graph = Graph(nodes=nodes, edges=edges, entry_point=entry_point or "")

    state_block = data.get("state", {})
    if isinstance(state_block, dict):
        schema = state_block.get("schema", {})
        initial = state_block.get("initial", {})
    else:
        schema = {}
        initial = {}

    reducers: dict[str, Reducer] = reducers_from_yaml_schema(schema)

    return graph, tools, initial, reducers


def _resolve_rag_config(config: dict, base_dir: str) -> dict:
    """Make relative paths in a ``rag`` tool config absolute.

    Resolves document file paths and ``store.path`` against *base_dir*
    (the directory of the workflow YAML).
    """

    def _abs(item: dict) -> dict:
        item = dict(item)
        for key in ("file", "path"):
            if key in item and not os.path.isabs(item[key]):
                item[key] = os.path.join(base_dir, item[key])
        return item

    result = dict(config)
    store = config.get("store")
    if isinstance(store, dict):
        result["store"] = _abs(store)

    docs = config.get("documents")
    if isinstance(docs, str):
        if not os.path.isabs(docs):
            result["documents"] = os.path.join(base_dir, docs)
    elif isinstance(docs, dict):
        result["documents"] = _abs(docs)
    elif isinstance(docs, list):
        result["documents"] = [
            _abs(doc) if isinstance(doc, dict) and "text" not in doc else doc
            for doc in docs
        ]
    return result
