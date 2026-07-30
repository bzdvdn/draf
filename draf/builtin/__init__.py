"""Built-in nodes registered by default."""

from draf.node.registry import default_registry
from draf.builtin.transform import Transform
from draf.builtin.llm import LLM

default_registry.register("transform", lambda cfg: Transform(cfg))
default_registry.register("llm_chat", lambda cfg: LLM(cfg))

__all__ = ["Transform", "LLM"]
