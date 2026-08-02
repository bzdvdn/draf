"""Context-building nodes — provided by the framework.

``ContextBuilder`` and ``AppendAssistant`` are the two built-in nodes from
:mod:`draf.node.context`, re-exported here so ``src.graphs.build`` reads the
same as a hand-written app, and ``last_user_message`` backs the supervisor
decider.

HOW TO EXTEND
    Add your own context builders as ``Node`` subclasses that compose
    ``state`` into the plain-text ``input`` your agents expect, then use
    them in ``graphs/build.py`` the same way.
"""

from draf.node.context import (  # noqa: F401
    AppendAssistant,
    ContextBuilder,
    last_user_message,
)
