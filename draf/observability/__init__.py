"""Graph-run observability: full traces (topology, node spans, LLM payloads).

Usage::

    from draf.observability import (
        GraphObserver,
        JsonlExporter,
        SQLiteExporter,
        topology_from_graph,
    )

    observer = GraphObserver(
        "my-flow",
        exporter=SQLiteExporter("./traces.db"),
        topology=topology_from_graph(graph),
    )
    await graph.run(state, tracer=observer.tracer,
                    on_llm_payload=observer.on_llm_payload)
    observer.export()
"""

from draf.observability.builder import (
    build_observability,
    build_observer_factory,
    build_remote_exporter,
)
from draf.observability.collector import GraphObserver
from draf.observability.exporter import (
    CompositeExporter,
    JsonlExporter,
    SQLiteExporter,
    TraceExporter,
)
from draf.observability.model import (
    GraphTopology,
    LLMCall,
    NodeSpan,
    Run,
    SpanEvent,
    ToolCall,
)
from draf.observability.push import HttpExporter, LangfuseExporter, LangsmithExporter
from draf.observability.topology import topology_from_graph

__all__ = [
    "GraphObserver",
    "CompositeExporter",
    "JsonlExporter",
    "SQLiteExporter",
    "TraceExporter",
    "HttpExporter",
    "LangfuseExporter",
    "LangsmithExporter",
    "build_observability",
    "build_observer_factory",
    "build_remote_exporter",
    "GraphTopology",
    "LLMCall",
    "NodeSpan",
    "Run",
    "SpanEvent",
    "ToolCall",
    "topology_from_graph",
]
