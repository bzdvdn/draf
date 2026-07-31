from draf.node.node import Node
from draf.node.registry import NodeRegistry, default_registry, node
from draf.node.context import ExecContext
from draf.node.retry import Retry
from draf.node.transform import Transform
from draf.node.llm import LLM
from draf.node.agent import ReActAgent, ToolExec
from draf.node.parallel import Parallel
from draf.node.map import Map

default_registry.register("transform", lambda cfg: Transform(cfg))
default_registry.register("llm_chat", lambda cfg: LLM(cfg))
default_registry.register("react_agent", lambda cfg: ReActAgent(cfg))
default_registry.register("tool_exec", lambda cfg: ToolExec(cfg))
default_registry.register(
    "parallel",
    lambda cfg: Parallel(cfg.get("branches", []) if isinstance(cfg, dict) else []),
)


def _map_factory(cfg: dict) -> Map:
    processor = cfg.get("processor", {})
    return Map(processor, config=cfg)


default_registry.register("map", _map_factory)
__all__ = [
    "Node",
    "NodeRegistry",
    "default_registry",
    "ExecContext",
    "node",
    "Retry",
    "Transform",
    "LLM",
    "ReActAgent",
    "ToolExec",
    "Parallel",
    "Map",
]
