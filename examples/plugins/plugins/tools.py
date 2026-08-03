"""Custom tool types for the plugins example.

Registered via the ``@tool`` decorator exactly like draf's built-ins, so
they show up in ``default_tool_registry`` and become usable both from
nodes (``ctx.tools[...]``) and from ReAct agents (``use_tools: all``).
"""

import re

from draf.tool.registry import tool


@tool("word_count", "Count the words in a text")
def word_count(text: str = "") -> str:
    """Return the number of whitespace-separated words in *text*."""
    return str(len([w for w in text.split() if w]))


@tool("slugify", "Convert a string to a lowercase URL slug")
def slugify(text: str = "") -> str:
    """Turn *text* into a URL-safe slug like ``hello-world``."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "-"
