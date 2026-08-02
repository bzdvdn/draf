"""Composition root — the single place that builds the app's object graph.

Re-exported so the rest of the app only needs ``from src.core import ...``.
"""

from src.core.container import Container, build_container

__all__ = ["Container", "build_container"]
