"""draf — workflow as data, agents as graphs.

A Python framework for building AI agent workflows using
graph-based pipelines with built-in tools and RAG support.
"""

from draf.node import (
    Node,
    NodeRegistry,
    default_registry,
    ExecContext,
    node,
    Retry,
    Transform,
    LLM,
    StructuredOutputError,
    ReActAgent,
    ToolExec,
    Parallel,
    Map,
    Interrupt,
    GraphInterrupt,
)
from draf.tool import Tool, ToolRegistry, default_tool_registry, tool
from draf.graph import Graph, Edge
from draf.flow import Flow, Case, SubFlow
from draf.harness import Harness, set_provider_concurrency, provider_concurrency
from draf.skill import Skill, load_skill, core_skills, get_core_skill
from draf.schema import json_schema_from_type, validate_json
from draf.trace import (
    RunTracer,
    TraceEvent,
    RunSummary,
    TokenUsage,
    model_pricing,
    tokens_cost,
    set_model_pricing,
    set_provider_pricing,
    load_pricing,
    clear_pricing,
)
from draf.yaml_schema import validate_workflow, validate_workflow_file
from draf.eval import run_eval, load_dataset, extract_output
from draf.rag import VectorStore, Embedder, Chunker, RAGTool, PDFTool, ImageTool
from draf.rag.stores import (
    InMemoryVectorStore,
    QdrantVectorStore,
    ChromaVectorStore,
    PGVectorStore,
)
from draf.yaml import from_yaml
from draf.state import (
    State,
    reducers_from_typeddict,
    reducers_from_yaml_schema,
    apply_reducers,
    Reducer,
)
from draf.stream import StreamEvent
from draf._version import __version__
from draf.errors import (
    DrafError,
    ConfigError,
    WorkflowError,
    NodeError,
    LLMError,
    InterruptError,
    redact,
)


def set_defaults(*, provider: str | None = None, **kwargs: object) -> None:
    """Set global defaults for the draf framework.

    Currently supported:
        provider — Default LLM provider (e.g. ``"ollama"``, ``"openai"``).
            Sets ``LLM.DEFAULT_PROVIDER`` so all LLM nodes use this
            provider unless overridden per-node.

    Usage::

        from draf import set_defaults
        set_defaults(provider="ollama")
    """
    if provider is not None:
        LLM.DEFAULT_PROVIDER = provider


__all__ = [
    "set_defaults",
    "__version__",
    "DrafError",
    "ConfigError",
    "WorkflowError",
    "NodeError",
    "LLMError",
    "InterruptError",
    "redact",
    "Node",
    "NodeRegistry",
    "default_registry",
    "ExecContext",
    "node",
    "Retry",
    "Tool",
    "ToolRegistry",
    "default_tool_registry",
    "tool",
    "Graph",
    "Edge",
    "Transform",
    "LLM",
    "StructuredOutputError",
    "ReActAgent",
    "ToolExec",
    "Parallel",
    "Map",
    "Interrupt",
    "GraphInterrupt",
    "Flow",
    "Case",
    "SubFlow",
    "Harness",
    "set_provider_concurrency",
    "provider_concurrency",
    "Skill",
    "load_skill",
    "core_skills",
    "get_core_skill",
    "json_schema_from_type",
    "validate_json",
    "VectorStore",
    "Embedder",
    "Chunker",
    "RAGTool",
    "PDFTool",
    "ImageTool",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    "PGVectorStore",
    "from_yaml",
    "State",
    "reducers_from_typeddict",
    "reducers_from_yaml_schema",
    "apply_reducers",
    "Reducer",
    "RunTracer",
    "TraceEvent",
    "RunSummary",
    "TokenUsage",
    "model_pricing",
    "tokens_cost",
    "set_model_pricing",
    "set_provider_pricing",
    "load_pricing",
    "clear_pricing",
    "validate_workflow",
    "validate_workflow_file",
    "run_eval",
    "load_dataset",
    "extract_output",
    "StreamEvent",
]
