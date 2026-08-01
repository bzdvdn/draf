from draf.tool.registry import default_tool_registry
from draf.tool.builtin.calculator import CalculatorTool
from draf.tool.builtin.shell import ShellTool
from draf.tool.builtin.file import ReadFileTool, WriteFileTool, EditFileTool
from draf.tool.builtin.web_search import WebSearchTool
from draf.tool.builtin.web_fetch import WebFetchTool
from draf.tool.builtin.pdf import PDFReadTool
from draf.tool.builtin.s3 import S3Tool, S3GetTool, S3PutTool
from draf.tool.builtin.slack import SlackSendTool
from draf.tool.builtin.sql import SQLQueryTool, SQLListTablesTool, SQLDescribeTool
from draf.tool.builtin.fs import ListDirTool, GlobTool, GetEnvTool, CurrentTimeTool
from draf.tool.builtin.data import (
    JsonParseTool,
    YamlParseTool,
    KVStoreTool,
    PythonEvalTool,
)
from draf.tool.builtin.http import HttpRequestTool
from draf.tool.builtin.notify import SendEmailTool, SendTelegramTool

default_tool_registry.register(CalculatorTool)
default_tool_registry.register(ShellTool)
default_tool_registry.register(ReadFileTool)
default_tool_registry.register(WriteFileTool)
default_tool_registry.register(EditFileTool)
default_tool_registry.register(WebSearchTool)
default_tool_registry.register(WebFetchTool)
default_tool_registry.register(PDFReadTool)
default_tool_registry.register(S3Tool)
default_tool_registry.register(S3GetTool)
default_tool_registry.register(S3PutTool)
default_tool_registry.register(SlackSendTool)
default_tool_registry.register(SQLQueryTool)
default_tool_registry.register(SQLListTablesTool)
default_tool_registry.register(SQLDescribeTool)
default_tool_registry.register(ListDirTool)
default_tool_registry.register(GlobTool)
default_tool_registry.register(GetEnvTool)
default_tool_registry.register(CurrentTimeTool)
default_tool_registry.register(JsonParseTool)
default_tool_registry.register(YamlParseTool)
default_tool_registry.register(KVStoreTool)
default_tool_registry.register(PythonEvalTool)
default_tool_registry.register(HttpRequestTool)
default_tool_registry.register(SendEmailTool)
default_tool_registry.register(SendTelegramTool)

__all__ = [
    "CalculatorTool",
    "ShellTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "WebSearchTool",
    "WebFetchTool",
    "PDFReadTool",
    "S3Tool",
    "S3GetTool",
    "S3PutTool",
    "SlackSendTool",
    "SQLQueryTool",
    "SQLListTablesTool",
    "SQLDescribeTool",
    "ListDirTool",
    "GlobTool",
    "GetEnvTool",
    "CurrentTimeTool",
    "JsonParseTool",
    "YamlParseTool",
    "KVStoreTool",
    "PythonEvalTool",
    "HttpRequestTool",
    "SendEmailTool",
    "SendTelegramTool",
]
