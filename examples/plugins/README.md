# Plugins — custom nodes & tools from a folder

Shows how to extend draf **without touching the framework**: drop a
`plugins/` folder (or a single `.py` file) next to your workflow, and the
custom node/tool types it registers become usable in YAML and Python
alike.

Layout::

    examples/plugins/
    ├── plugins/
    │   ├── nodes.py          # custom node types via @node
    │   ├── tools.py          # custom tool types via @tool
    │   └── classes.py        # the same, using subclasses (no decorators)
    ├── workflow.yaml         # offline pipeline (no LLM)
    ├── workflow-agent.yaml   # ReAct agent using the custom tools (Ollama)
    ├── workflow-classes.yaml # class-based plugin, offline
    └── run.py                # runs workflow.yaml programmatically

## How it works

A plugin is any Python module that registers types with draf — *loading
the file is the whole mechanism*.  There are two equivalent styles:

**Decorators** (`nodes.py` / `tools.py`):

```python
from draf.node.registry import node
from draf.tool.registry import tool

@node("slugify_node", SlugConfig)
async def slugify_node(ctx, config, state):
    ...

@tool("slugify", "Convert a string to a lowercase URL slug")
def slugify(text: str = "") -> str:
    ...
```

**Subclasses** (`classes.py`):

```python
from draf.node.node import Node
from draf.node.registry import default_registry
from draf.tool.tool import Tool
from draf.tool.registry import default_tool_registry

class UpperTool(Tool):
    name = "upper"
    description = "Uppercase a string"

    def run(self, text: str = "") -> str:
        return text.upper()

class UppercaseNode(Node):
    type = "uppercase_node"

    async def execute(self, ctx, state):
        ...

default_tool_registry.register(UpperTool)
default_registry.register("uppercase_node", UppercaseNode)
```

Both register on import (decorator) or explicitly (class), so they land in
the same shared registries. `load_workflow` / `draf validate` discover the
files two ways:

1. **Default folder**: any `plugins/` directory next to the workflow is
   auto-loaded (every `*.py` except `_`-prefixed helpers).  Override the
   location with the `plugins_folder:` key.
2. **Explicit key**: list files/folders under `plugins:` in the YAML.

`workflow.yaml` here relies on the default folder — no `plugins:` key
needed.  The explicit form would look like:

```yaml
plugins:
  - plugins/nodes.py
  - plugins/tools.py
```

To point at a different folder (e.g. vendored plugins elsewhere):

```yaml
plugins_folder: vendor/draf-plugins
```

The loaded types land in the shared registries, so validation
(`draf validate workflow.yaml`) accepts them and they can be referenced
in `steps:` / `tools:` exactly like built-ins.

> Tools are only visible to a run if they are listed under `tools:` in
> the workflow (or passed to `graph.run(tools=...)`).  Custom *node* types
> that call a tool through `ctx.tools` therefore need that tool declared —
> `workflow.yaml` declares `slugify` and `word_count` for this reason.

## What the example does

`plugins/tools.py` adds two pure-Python tools, `word_count` and `slugify`.
`plugins/nodes.py` adds three node types:

- `slugify_node` and `word_count_node` — call the plugin tools through
  `ctx.tools`, showing how a node can use tools directly (no LLM needed);
- `format_json` — pretty-prints a slice of state, using a typed config
  dataclass.

The pipeline in `workflow.yaml` chains them: slugify the body, count its
words, then render a JSON report — fully offline.

## Run

```bash
# offline, no LLM
python examples/plugins/run.py
uv run draf validate examples/plugins/workflow.yaml

# class-based plugin, offline
uv run draf daemon -f workflow-classes.yaml --once

# the agent variant needs Ollama with llama3.1:8b
ollama pull llama3.1:8b
uv run draf daemon -f workflow-agent.yaml --once
```
