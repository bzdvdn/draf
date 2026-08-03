from draf.tool.builtin.calculator import CalculatorTool
from draf.tool.builtin.csv import CsvQueryTool
from draf.tool.builtin.data import (
    JsonParseTool,
    KVStoreTool,
    PythonEvalTool,
    YamlParseTool,
)
from draf.tool.builtin.file import EditFileTool, ReadFileTool, WriteFileTool
from draf.tool.builtin.fs import CurrentTimeTool, GetEnvTool, GlobTool, ListDirTool
from draf.tool.builtin.git import GitTool
from draf.tool.builtin.github import (
    GitHubApproveTool,
    GitHubGetPRChangesTool,
    GitHubListOpenPRsTool,
    GitHubPostCommentTool,
)
from draf.tool.builtin.gitlab import (
    GitLabApproveTool,
    GitLabGetMRChangesTool,
    GitLabListOpenMRsTool,
    GitLabPostNoteTool,
)
from draf.tool.builtin.http import HttpRequestTool
from draf.tool.builtin.lock import LockTool
from draf.tool.builtin.notify import SendEmailTool, SendTelegramTool
from draf.tool.builtin.pdf import PDFReadTool
from draf.tool.builtin.redis import RedisTool
from draf.tool.builtin.s3 import S3GetTool, S3PutTool, S3Tool
from draf.tool.builtin.shell import ShellTool
from draf.tool.builtin.slack import SlackSendTool
from draf.tool.builtin.sql import SQLDescribeTool, SQLListTablesTool, SQLQueryTool
from draf.tool.builtin.wait_for import WaitForTool
from draf.tool.builtin.web_fetch import WebFetchTool
from draf.tool.builtin.web_search import WebSearchTool
from draf.tool.registry import default_tool_registry

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
default_tool_registry.register(RedisTool)
default_tool_registry.register(GitTool)
default_tool_registry.register(LockTool)
default_tool_registry.register(WaitForTool)
default_tool_registry.register(CsvQueryTool)
default_tool_registry.register(GitLabListOpenMRsTool)
default_tool_registry.register(GitLabGetMRChangesTool)
default_tool_registry.register(GitLabPostNoteTool)
default_tool_registry.register(GitLabApproveTool)
default_tool_registry.register(GitHubListOpenPRsTool)
default_tool_registry.register(GitHubGetPRChangesTool)
default_tool_registry.register(GitHubPostCommentTool)
default_tool_registry.register(GitHubApproveTool)

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
    "RedisTool",
    "GitTool",
    "LockTool",
    "WaitForTool",
    "CsvQueryTool",
    "GitLabListOpenMRsTool",
    "GitLabGetMRChangesTool",
    "GitLabPostNoteTool",
    "GitLabApproveTool",
    "GitHubListOpenPRsTool",
    "GitHubGetPRChangesTool",
    "GitHubPostCommentTool",
    "GitHubApproveTool",
]
