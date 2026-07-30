from draf.tool.registry import default_tool_registry
from draf.tool.builtin.calculator import CalculatorTool
from draf.tool.builtin.shell import ShellTool
from draf.tool.builtin.file import ReadFileTool, WriteFileTool, EditFileTool
from draf.tool.builtin.web_search import WebSearchTool

default_tool_registry.register(CalculatorTool)
default_tool_registry.register(ShellTool)
default_tool_registry.register(ReadFileTool)
default_tool_registry.register(WriteFileTool)
default_tool_registry.register(EditFileTool)
default_tool_registry.register(WebSearchTool)

__all__ = [
    "CalculatorTool", "ShellTool",
    "ReadFileTool", "WriteFileTool", "EditFileTool",
    "WebSearchTool",
]
