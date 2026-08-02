"""Composition root — build the whole application object graph once.

``build_container`` is the single place that constructs the durable assets
(graph, tools, checkpointer, assistant) from
:class:`~src.config.config.Settings`.  Every entry point (``app.py`` /
``cli.py`` / ``daemon.py``) goes through it, so session hydration, tool
wiring and storage are identical across interfaces.

When the ``rag`` variant is installed, :func:`build_container` also builds a
document catalog (from ``data/documents/``) and hands it to the graph, so the
agents can search indexed documents out of the box.

HOW TO EXTEND
    * Swap the LLM provider/model: set ``DRAF_PROVIDER`` / ``DRAF_MODEL``.
    * Add a tool: build it in ``src/tools`` and wire it in here.
    * Swap storage: see ``src.storage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.config import Settings, get_settings
from src.graphs.build import build_flow
from src.service.assistant import Assistant
from src.storage import build_checkpointer


@dataclass
class Container:
    """The application object graph: graph, tools, checkpointer, assistant.

    Attributes:
        settings: The resolved settings this container was built from.
        graph: The compiled supervisor graph.
        tools: The tool set handed to the graph (base tools + ``rag``).
        checkpointer: The durable session backend.
        catalog: The document catalog when the ``rag`` variant is present,
            else ``None``.
        assistant: The :class:`~src.service.assistant.Assistant` service.
    """

    settings: Settings
    graph: Any
    tools: list
    checkpointer: Any
    catalog: Any = None
    assistant: Assistant = field(init=False)

    def __post_init__(self) -> None:
        self.assistant = Assistant(self.graph, self.tools, self.checkpointer)


def _build_catalog(settings: Settings):
    """Return the document catalog when the ``rag`` variant is present.

    The catalog wiring module only exists in rag-enabled projects, so a
    plain project gets ``None`` (no import, no feature).  Building the
    catalog never touches the network: documents are embedded lazily on the
    first search.
    """
    try:
        from src.rag.wiring import build_catalog
    except ImportError:
        return None
    return build_catalog(settings)


def build_container(
    settings: Settings | None = None,
    *,
    checkpoint_dir: str | None = None,
) -> Container:
    """Build the durable app assets from *settings* (environment by default).

    Args:
        settings: Override the environment settings (tests inject one).
        checkpoint_dir: Convenience override for the checkpoint location;
            copied onto *settings* when given.
    """
    settings = settings or get_settings()
    if checkpoint_dir is not None:
        settings = settings.model_copy(update={"checkpoint_dir": checkpoint_dir})
    checkpointer = build_checkpointer(
        settings.checkpoint_dir, checkpoint_db=settings.checkpoint_db
    )
    catalog = _build_catalog(settings)
    flow, tools = build_flow(
        model=settings.model, provider=settings.provider, catalog=catalog
    )
    return Container(
        settings=settings,
        graph=flow.compile(),
        tools=tools,
        checkpointer=checkpointer,
        catalog=catalog,
    )
