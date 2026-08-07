"""Class-based plugin (no decorators).

The same plugin mechanism as ``nodes.py`` / ``tools.py``, but written with
plain subclasses instead of ``@tool`` / ``@node``:

- a ``Tool`` subclass sets ``name`` / ``description`` and implements
  ``run``, then is added to ``default_tool_registry``;
- a ``Node`` subclass sets ``type`` and implements
  ``execute(ctx, state)``, then is added to ``default_registry`` under a
  type name.

Both registrations are *import side effects*, exactly like the decorator
form, so this file is picked up by the same folder discovery.

Here ``UppercaseNode`` calls the class-based ``UpperTool`` through
``ctx.tools`` and reads its config from ``self.config``.
"""

from teff.node.node import Node
from teff.node.registry import default_registry
from teff.tool.registry import default_tool_registry
from teff.tool.tool import Tool


class UpperTool(Tool):
    name = "upper"
    description = "Uppercase a string"

    def run(self, text: str = "") -> str:  # type: ignore[override]
        return text.upper()


class UppercaseNode(Node):
    type = "uppercase_node"

    async def execute(self, ctx, state: dict) -> dict:
        input_key = self.config.get("input_key", "text")
        output_key = self.config.get("output_key", "out")
        state[output_key] = ctx.tools["upper"].run(text=state.get(input_key, ""))
        return state


default_tool_registry.register(UpperTool)
default_registry.register("uppercase_node", UppercaseNode)
