"""YAML serialisation and deserialisation for graphs."""

import os
import re
import typing

import yaml

from teff.errors import ConfigError
from teff.graph import Edge, Graph
from teff.node.node import Node
from teff.state.state import Reducer, reducers_from_yaml_schema
from teff.tool.registry import default_tool_registry
from teff.tool.tool import Tool
from teff.yaml_schema import raise_for_validation, validate_workflow


def _safe_load(source: typing.Any) -> typing.Any:
    """Load YAML, surfacing parse errors as ``ConfigError``."""
    try:
        return yaml.safe_load(source)
    except (yaml.YAMLError, ValueError) as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc


_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env(value: typing.Any) -> typing.Any:
    """Replace ``${ENV_VAR}`` references in a YAML structure with values.

    Recursively walks strings inside mappings and lists.  A reference to a
    missing variable is left unchanged (rather than raising) so templates
    stay valid offline; set the variable to inject the secret.
    """
    if isinstance(value, str):
        return _ENV_VAR.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _providers_from_data(data: typing.Any) -> typing.Any:
    """Build a :class:`ProviderRegistry` from a workflow's ``providers:`` block.

    The block is a list of mappings, each with a ``name:`` plus any provider
    fields.  A ``name`` that matches a built-in preset is merged over that
    preset's defaults — ``{name: ollama}`` alone yields the ready-made Ollama
    endpoint, and ``{name: ollama, base_url: http://remote:11434}`` overrides
    just the endpoint while keeping the rest.  Names that are not presets are
    plain custom providers spelled out entirely (e.g. ``{name: my-vllm,
    type: openai_compatible, base_url: http://vllm:8000/v1}``).  Bare
    preset-name strings are rejected — only mappings are accepted.  Each
    ``name`` must be unique (a duplicate raises :class:`ConfigError`).
    Returns ``None`` when the block is absent.
    """
    from teff.provider import (
        BUILTINS,
        PROVIDER_FIELDS,
        Provider,
        ProviderRegistry,
    )

    block = data.get("providers")
    if not block:
        return None
    if not isinstance(block, list):
        raise ConfigError("providers must be a list")
    reg = ProviderRegistry()
    for entry in block:
        if isinstance(entry, str):
            raise ConfigError(
                f"providers: preset names are not allowed ({entry!r}) — "
                "use a `{name, ...}` mapping (a built-in name merges its preset)"
            )
        if not isinstance(entry, dict):
            raise ConfigError("providers: each entry must be a mapping with a `name:`")
        name = entry.get("name")
        if not name:
            raise ConfigError("providers: each mapping entry requires a `name:`")
        unknown = set(entry) - set(PROVIDER_FIELDS)
        if unknown:
            raise ConfigError(
                f"providers.{name} has unknown keys: {', '.join(sorted(unknown))}"
            )
        preset = BUILTINS.get(name)
        if preset is not None:
            fields = preset().to_dict()
            fields.update({f: entry[f] for f in PROVIDER_FIELDS if f in entry})
            reg.register(Provider.from_mapping(fields))
        else:
            reg.register(Provider.from_mapping(entry))
    return reg


def _validate_provider_refs(data: typing.Any, registry: typing.Any) -> None:
    """Enforce every provider reference in *data* is declared.

    ``default_provider:`` and each step's ``config.provider`` must name a
    provider declared in the ``providers:`` block — there is no implicit
    built-in fallback.
    """
    valid = set(registry or ())
    dp = data.get("default_provider")
    if dp and dp not in valid:
        raise ConfigError(f"default_provider {dp!r} is not declared in `providers:`")
    for step in data.get("steps", []):
        prov = (step.get("config") or {}).get("provider")
        sid = step.get("id")
        if prov and prov not in valid:
            raise ConfigError(
                f"step {sid!r}: provider {prov!r} is not declared in `providers:`"
            )


def _providers_to_block(registry: typing.Any) -> list:
    """Serialize ``graph.providers`` back into a ``providers:`` list.

    Every provider is written as a ``{name, ...}`` mapping — never a bare
    string, so the file always spells out what is configured.
    """
    return [provider.to_dict() for name, provider in registry.items()]


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

    from teff.node.registry import default_registry

    nodes = {}
    edges = []
    entry_point = None

    for step in data.get("steps", []):
        sid = step["id"]
        stype = step["type"]
        config = step.get("config", {})
        node = default_registry.create(stype, config)
        if step.get("retry"):
            from teff.node.retry import wrap_with_retry

            node = wrap_with_retry(node, step["retry"])
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

    providers = _providers_from_data(data)
    _validate_provider_refs(data, providers)

    return Graph(
        nodes=nodes,
        edges=edges,
        entry_point=entry_point or "",
        providers=providers,
        default_provider=data.get("default_provider"),
        default_model=data.get("default_model"),
    )


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
        if callable(e.condition):
            raise ValueError(
                "cannot serialize a callable edge condition to YAML "
                "(callable conditions are programmatic-only; use a string "
                f"condition or a decider key for the edge {e.source_id!r} -> {e.target_id!r})"
            )
        entry = {"from": e.source_id, "to": e.target_id}
        if e.condition:
            entry["condition"] = e.condition
        edges.append(entry)

    data: dict = {"name": name, "steps": steps, "edges": edges}
    if getattr(graph, "default_provider", None):
        data["default_provider"] = graph.default_provider
    if getattr(graph, "default_model", None):
        data["default_model"] = graph.default_model
    if getattr(graph, "providers", None) and len(graph.providers):
        data["providers"] = _providers_to_block(graph.providers)
    if tools:
        data["tools"] = [{"type": t.name, "config": _tool_config(t)} for t in tools]
    if reducers or initial:
        from teff.state.state import reducers_to_yaml_schema

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
    from teff.plugins import load_plugins_from_document

    load_plugins_from_document(data, os.path.dirname(os.path.abspath(path)))

    import teff.rag  # noqa: F401 — registers the "rag" tool
    import teff.tool.builtin  # noqa: F401 — registers built-in tools

    raise_for_validation(validate_workflow(data), source=path)

    from teff.node.registry import default_registry

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    entry_point: str | None = None

    for step in data.get("steps", []):
        sid = step["id"]
        stype = step["type"]
        config = step.get("config", {})
        node = default_registry.create(stype, config)
        if step.get("retry"):
            from teff.node.retry import wrap_with_retry

            node = wrap_with_retry(node, step["retry"])
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

    providers = _providers_from_data(data)
    _validate_provider_refs(data, providers)

    graph = Graph(
        nodes=nodes,
        edges=edges,
        entry_point=entry_point or "",
        providers=providers,
        default_provider=data.get("default_provider"),
        default_model=data.get("default_model"),
    )

    state_block = data.get("state", {})
    if isinstance(state_block, dict):
        schema = state_block.get("schema", {})
        initial = state_block.get("initial", {})
    else:
        schema = {}
        initial = {}

    if not isinstance(initial, dict):
        raise ConfigError("state.initial must be a mapping")
    if schema:
        from teff.state.state import validate_state

        errors = validate_state(initial, schema)
        if errors:
            raise ConfigError(
                "state.initial does not match state.schema:\n"
                + "\n".join(f"  {e}" for e in errors)
            )

    reducers: dict[str, Reducer] = reducers_from_yaml_schema(schema)

    return graph, tools, initial, reducers


def checkpointer_from_workflow(path: str):
    """Build the checkpointer declared by a workflow's ``checkpoint:`` block.

    Returns ``None`` when the workflow has no ``checkpoint:`` block.
    Relative ``path`` values are resolved against the workflow file's
    directory; ``dsn`` values are passed through verbatim.
    """
    from teff.checkpoint.from_config import (
        checkpointer_from_config,
        resolve_checkpoint_config,
    )

    with open(path) as f:
        data = _safe_load(f)
    if not isinstance(data, dict):
        return None
    if "checkpoint" not in data:
        return None
    base_dir = os.path.dirname(os.path.abspath(path))
    return checkpointer_from_config(
        resolve_checkpoint_config(data["checkpoint"], base_dir)
    )


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
