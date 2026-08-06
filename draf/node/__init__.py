from typing import TYPE_CHECKING

from draf.node.agent import ReActAgent, ToolExec
from draf.node.ask import Ask, Validate

if TYPE_CHECKING:
    from draf.provider import ProviderRegistry
from draf.node.context import (
    AppendAssistant,
    ContextBuilder,
    ExecContext,
    last_user_message,
)
from draf.node.gate import Gate
from draf.node.interrupt import GraphInterrupt, Interrupt
from draf.node.llm import LLM, StructuredOutputError
from draf.node.map import Map
from draf.node.node import Node
from draf.node.parallel import Parallel
from draf.node.registry import NodeRegistry, default_registry, node
from draf.node.retry import Retry
from draf.node.supervisor import Supervisor
from draf.node.tool_call import ToolCall
from draf.node.transform import Transform

if TYPE_CHECKING:
    from draf.flow.sub_flow import SubFlow
    from draf.graph import Graph

default_registry.register("transform", lambda cfg: Transform(cfg))
default_registry.register("gate", lambda cfg: Gate(cfg))
default_registry.register("validate", lambda cfg: Validate(cfg))
default_registry.register("context_builder", lambda cfg: ContextBuilder(cfg))
default_registry.register("append_assistant", lambda cfg: AppendAssistant(cfg))
default_registry.register("llm_chat", lambda cfg: LLM(cfg))
default_registry.register("react_agent", lambda cfg: ReActAgent(cfg))
default_registry.register("tool_exec", lambda cfg: ToolExec(cfg))
default_registry.register("tool_call", lambda cfg: ToolCall(cfg))
default_registry.register("interrupt", lambda cfg: Interrupt(cfg))
default_registry.register("supervisor", lambda cfg: Supervisor(cfg))
default_registry.register(
    "parallel",
    lambda cfg: Parallel(cfg.get("branches", []) if isinstance(cfg, dict) else []),
)


def _map_factory(cfg: dict) -> Map:
    processor = cfg.get("processor", {})
    return Map(processor, config=cfg)


def _subflow_factory(cfg: dict) -> "SubFlow":
    from draf.errors import ConfigError
    from draf.flow.sub_flow import SubFlow

    graph_cfg = cfg.get("graph")
    build = cfg.get("build")
    if isinstance(graph_cfg, dict):
        flow = SubFlow(
            _build_subgraph(graph_cfg, cfg.get("providers")),
            input_map=cfg.get("input_map"),
            output_map=cfg.get("output_map"),
            max_iterations=cfg.get("max_iterations"),
            id_prefix=cfg.get("id_prefix", ""),
        )
    elif build is not None:
        flow = _build_from_recipe(build, cfg)
    else:
        raise ConfigError(
            "subflow requires config.graph (a mapping with steps/edges) "
            "or config.build (an agent_step recipe)"
        )
    flow.config = cfg
    return flow


def _build_subgraph(
    graph_cfg: dict, providers: "dict | ProviderRegistry | None" = None
) -> "Graph":
    """Build a nested ``Graph`` from a declarative ``{steps, edges}`` dict.

    Mirrors the step/edge building in :mod:`draf.yaml` so a ``subflow``
    node can embed a full graph inline and round-trip through YAML.
    """
    from draf.errors import ConfigError
    from draf.graph import Edge, Graph

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    entry_point: str | None = None

    for step in graph_cfg.get("steps", []):
        if not isinstance(step, dict):
            raise ConfigError("subflow graph steps must be mappings")
        sid = step.get("id")
        stype = step.get("type")
        if not isinstance(sid, str) or not isinstance(stype, str):
            raise ConfigError("subflow graph step requires string id and type")
        node = default_registry.create(stype, step.get("config", {}))
        if step.get("retry"):
            from draf.node.retry import wrap_with_retry

            node = wrap_with_retry(node, step["retry"])
        nodes[sid] = node
        if entry_point is None:
            entry_point = sid

    for edge_data in graph_cfg.get("edges", []):
        if not isinstance(edge_data, dict):
            raise ConfigError("subflow graph edges must be mappings")
        edges.append(
            Edge(
                source_id=edge_data["from"],
                target_id=edge_data["to"],
                condition=edge_data.get("condition"),
            )
        )

    return Graph(
        nodes=nodes,
        edges=edges,
        entry_point=entry_point or "",
        providers=providers,
    )


def _build_from_recipe(build: dict, cfg: dict) -> "SubFlow":
    """Build a ``SubFlow`` from a named recipe such as ``agent_step``."""
    from draf.errors import ConfigError
    from draf.flow.agent import agent_step

    rtype = build.get("type")
    if rtype != "agent_step":
        raise ConfigError(
            f"unknown subflow build recipe {rtype!r} (supported: agent_step)"
        )
    if build.get("providers"):
        raise ConfigError(
            "agent_step build recipe must not set `providers:` — providers "
            "come from the workflow's top-level `providers:` block"
        )
    try:
        return agent_step(
            build["system"],
            build["output_key"],
            model=build["model"],
            provider=build["provider"],
            sections=build.get("sections"),
            messages_key=build.get("messages_key", "messages"),
            use_tools=build.get("use_tools"),
            stream=build.get("stream", True),
            id=cfg.get("id_prefix") or None,
            **build.get("config", {}),
        )
    except KeyError as exc:
        raise ConfigError(
            f"agent_step build recipe is missing required key: {exc.args[0]}"
        ) from exc


default_registry.register("map", _map_factory)
default_registry.register("subflow", _subflow_factory)
__all__ = [
    "Node",
    "NodeRegistry",
    "default_registry",
    "ExecContext",
    "ContextBuilder",
    "AppendAssistant",
    "last_user_message",
    "node",
    "Retry",
    "Transform",
    "LLM",
    "StructuredOutputError",
    "ReActAgent",
    "ToolExec",
    "ToolCall",
    "Parallel",
    "Map",
    "Supervisor",
    "Interrupt",
    "GraphInterrupt",
    "Ask",
    "Validate",
]
