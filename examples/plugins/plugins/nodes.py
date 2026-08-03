"""Custom node types for the plugins example.

Everything in this folder is a **plugin**: a plain Python module that
registers new node and tool types with draf's existing ``@node`` / ``@tool``
decorators.  Because registration happens on import, loading the file is
all that is needed to make the new types visible to ``draf validate`` and
``load_workflow``.

The folder is discovered automatically — a workflow sitting next to a
``plugins/`` directory picks these up with no YAML changes (or you can
list files/folders explicitly under ``plugins:``).

Here we define three node types.  ``format_json`` is a plain transform;
``slugify_node`` and ``word_count_node`` show a node calling a *custom*
tool (registered in ``tools.py``) through ``ctx.tools``.
"""

import json
from dataclasses import dataclass, field

from draf.node.registry import node


@dataclass
class SlugConfig:
    input_key: str = "text"
    output_key: str = "slug"


@node("slugify_node", SlugConfig)
async def slugify_node(ctx, config: SlugConfig, state):
    """Slugify ``state[config.input_key]`` using the plugin's ``slugify`` tool."""
    tool = ctx.tools["slugify"]
    state[config.output_key] = tool.run(text=state.get(config.input_key, ""))
    return state


@dataclass
class CountConfig:
    input_key: str = "text"
    output_key: str = "count"


@node("word_count_node", CountConfig)
async def word_count_node(ctx, config: CountConfig, state):
    """Count words using the plugin's ``word_count`` tool."""
    tool = ctx.tools["word_count"]
    state[config.output_key] = tool.run(text=state.get(config.input_key, ""))
    return state


@dataclass
class FormatConfig:
    keys: list = field(default_factory=list)
    output_key: str = "report"


@node("format_json", FormatConfig)
async def format_json(ctx, config: FormatConfig, state):
    """Pretty-print a slice of the state as JSON."""
    report = {key: state.get(key) for key in config.keys}
    state[config.output_key] = json.dumps(report, indent=2, ensure_ascii=False)
    return state
