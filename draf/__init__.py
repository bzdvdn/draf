"""draf — workflow as data, agents as graphs.

A Python framework for building AI agent workflows using
graph-based pipelines with built-in tools and RAG support.
"""

from draf._version import __version__
from draf.errors import (
    ConfigError,
    DrafError,
    InterruptError,
    LLMError,
    NodeError,
    WorkflowError,
    redact,
)
from draf.eval import extract_output, load_dataset, run_eval
from draf.flow import Case, Flow, SubFlow
from draf.graph import Edge, Graph
from draf.harness import Harness, provider_concurrency, set_provider_concurrency
from draf.node import (
    LLM,
    ExecContext,
    GraphInterrupt,
    Interrupt,
    Map,
    Node,
    NodeRegistry,
    Parallel,
    ReActAgent,
    Retry,
    StructuredOutputError,
    ToolExec,
    Transform,
    default_registry,
    node,
)
from draf.rag import Chunker, Embedder, ImageTool, PDFTool, RAGTool, VectorStore
from draf.rag.stores import (
    ChromaVectorStore,
    InMemoryVectorStore,
    PGVectorStore,
    QdrantVectorStore,
)
from draf.schema import json_schema_from_type, validate_json
from draf.skill import Skill, core_skills, get_core_skill, load_skill
from draf.state import (
    Reducer,
    State,
    apply_reducers,
    reducers_from_typeddict,
    reducers_from_yaml_schema,
    state_schema_to_jsonschema,
    validate_state,
)
from draf.stream import StreamEvent
from draf.tool import Tool, ToolRegistry, default_tool_registry, tool
from draf.trace import (
    RunSummary,
    RunTracer,
    TokenUsage,
    TraceEvent,
    clear_pricing,
    load_pricing,
    model_pricing,
    set_model_pricing,
    set_provider_pricing,
    tokens_cost,
)
from draf.yaml import from_yaml
from draf.yaml_schema import validate_workflow, validate_workflow_file


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
    "state_schema_to_jsonschema",
    "validate_state",
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
