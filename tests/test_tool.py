import pytest


class TestTool:
    def test_subclass_with_run(self):
        from draf.tool import Tool

        class MyTool(Tool):
            name = "mytool"
            description = "Does stuff"

            def run(self, x: int = 0) -> int:  # type: ignore[override]
                return x * 2

        t = MyTool()
        assert t.name == "mytool"
        assert t.description == "Does stuff"
        assert t.run(x=21) == 42

    def test_arun_falls_back_to_run(self):
        import asyncio

        from draf.tool import Tool

        class MyTool(Tool):
            name = "calc"
            description = "calc"

            def run(self, x: int = 0) -> int:  # type: ignore[override]
                return x + 1

        t = MyTool()
        r = asyncio.run(t.arun(x=41))
        assert r == 42


class TestCoerceArgs:
    def test_coerces_strings_to_declared_types(self):
        from draf.tool import Tool
        from draf.tool.tool import coerce_args

        class MyTool(Tool):
            name = "typed"
            description = "typed"

            def run(self, count: int = 0, ratio: float = 0.0, flag: bool = False):  # type: ignore[override]
                return (count, ratio, flag)

        t = MyTool()
        args = coerce_args(t, {"count": "3", "ratio": "1.5", "flag": "true"})
        assert args == {"count": 3, "ratio": 1.5, "flag": True}

    def test_coerces_against_async_arun(self):
        from draf.tool import Tool
        from draf.tool.tool import coerce_args

        class MyTool(Tool):
            name = "asynct"
            description = "asynct"

            async def arun(self, k: int = 5) -> int:  # type: ignore[override]
                return k

        t = MyTool()
        args = coerce_args(t, {"k": "1"})
        assert args == {"k": 1}

    def test_leaves_correct_types_untouched(self):
        from draf.tool import Tool
        from draf.tool.tool import coerce_args

        class MyTool(Tool):
            name = "already"
            description = "already"

            def run(self, n: int = 0):  # type: ignore[override]
                return n

        t = MyTool()
        args = coerce_args(t, {"n": 7})
        assert args == {"n": 7}

    def test_coerces_optional_float(self):
        from draf.tool import Tool
        from draf.tool.tool import coerce_args

        class MyTool(Tool):
            name = "optfloat"
            description = "optfloat"

            async def arun(  # type: ignore[override]
                self, query: str = "", max_price: float | None = None
            ) -> str:
                return f"{query}:{max_price}"

        t = MyTool()
        args = coerce_args(t, {"query": "дверь", "max_price": "2000"})
        assert args == {"query": "дверь", "max_price": 2000.0}
        assert isinstance(args["max_price"], float)

    def test_leaves_optional_none_untouched(self):
        from draf.tool import Tool
        from draf.tool.tool import coerce_args

        class MyTool(Tool):
            name = "optnone"
            description = "optnone"

            async def arun(  # type: ignore[override]
                self, max_price: float | None = None
            ) -> str:
                return str(max_price)

        t = MyTool()
        args = coerce_args(t, {"max_price": None})
        assert args == {"max_price": None}

    def test_unwraps_typing_optional(self):
        import typing

        from draf.tool import Tool
        from draf.tool.tool import coerce_args

        class MyTool(Tool):
            name = "typingopt"
            description = "typingopt"

            async def arun(  # type: ignore[override]
                self,
                count: typing.Optional[int] = None,  # noqa: UP045
            ) -> str:
                return str(count)

        t = MyTool()
        args = coerce_args(t, {"count": "7"})
        assert args == {"count": 7}


class TestToolDecorator:
    def test_registers_and_creates(self):
        from draf.tool import default_tool_registry, tool

        @tool("hello", description="Says hello")
        def greet(name: str = "") -> str:
            return f"Hello {name}"

        assert "hello" in default_tool_registry.list()
        t = default_tool_registry.create("hello")
        assert t.name == "hello"
        assert t.description == "Says hello"
        assert t.run(name="World") == "Hello World"

    def test_async_tool(self):
        import asyncio

        from draf.tool import default_tool_registry, tool

        @tool("async_hello")
        async def greet(name: str = "") -> str:
            return f"Hi {name}"

        t = default_tool_registry.create("async_hello")
        r = asyncio.run(t.arun(name="Test"))
        assert r == "Hi Test"


class TestToolRegistry:
    def test_register_and_list(self):
        from draf.tool import Tool, ToolRegistry

        class FT(Tool):
            name = "ft"
            description = "ft"

            def run(self):
                return "ok"

        reg = ToolRegistry()
        reg.register(FT)
        assert "ft" in reg.list()
        t = reg.create("ft")
        assert t.run() == "ok"

    def test_create_passes_config_dict_to_config_constructor(self):
        from draf.tool import ToolRegistry
        from draf.tool.builtin import SQLQueryTool

        reg = ToolRegistry()
        reg.register(SQLQueryTool)
        t = reg.create("sql_query", {"db_type": "sqlite", "path": "x.db"})
        assert t.db_type == "sqlite"
        assert t.path == "x.db"

    def test_create_passes_config_as_kwargs_to_keyword_constructor(self):
        from draf.tool import ToolRegistry
        from draf.tool.builtin import ShellTool, WebSearchTool

        reg = ToolRegistry()
        reg.register(ShellTool)
        reg.register(WebSearchTool)
        shell = reg.create("shell", {"root_dir": "/tmp", "allowed_commands": ["echo"]})
        assert shell.root_dir == "/tmp"
        assert shell._allowed == ["echo"]
        search = reg.create("web_search", {"provider": "google"})
        assert search.provider == "google"


class TestBuiltinTools:
    def test_calculator(self):
        from draf.tool.builtin import CalculatorTool

        t = CalculatorTool()
        assert t.run(expression="2+2") == "4"
        assert t.run(expression="3*4") == "12"
        assert t.run(expression="10/2") == "5.0"

    @pytest.mark.asyncio
    async def test_shell(self):
        from draf.tool.builtin import ShellTool

        t = ShellTool()
        r = await t.arun(command="echo ok")
        assert "ok" in r

    def test_file_tools(self, tmp_path):
        from draf.tool.builtin.file import EditFileTool, ReadFileTool, WriteFileTool

        path = str(tmp_path / "test.txt")
        wt = WriteFileTool()
        wt.run(path=path, content="hello world")
        rt = ReadFileTool()
        assert rt.run(path=path) == "hello world"
        et = EditFileTool()
        et.run(path=path, old="world", new="there")
        assert rt.run(path=path) == "hello there"


class TestExtendedBuiltinTools:
    @pytest.mark.asyncio
    async def test_web_fetch_empty_url(self):
        from draf.tool.builtin import WebFetchTool

        with pytest.raises(ValueError, match="url"):
            await WebFetchTool().arun()

    @pytest.mark.asyncio
    async def test_web_fetch_missing_bs4(self, monkeypatch):
        import builtins

        import httpx

        from draf.tool.builtin import WebFetchTool

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "bs4":
                raise ImportError("no beautifulsoup4")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        class FakeResponse:
            text = "<html><body>hello <b>world</b></body></html>"

            def raise_for_status(self):
                pass

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, headers=None):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())
        with pytest.raises(ImportError, match="beautifulsoup4"):
            await WebFetchTool().arun(url="http://example.com")

    def test_pdf_read_requires_path(self):
        from draf.tool.builtin import PDFReadTool

        with pytest.raises(ValueError, match="path"):
            PDFReadTool().run()

    def test_pdf_read_missing_pypdf(self, monkeypatch):
        import builtins

        from draf.tool.builtin import PDFReadTool

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf":
                raise ImportError("no pypdf")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            with pytest.raises(ImportError, match="pypdf"):
                PDFReadTool().run(path=f.name)

    def test_s3_list_requires_bucket(self):
        from draf.tool.builtin import S3Tool

        with pytest.raises(ValueError, match="bucket"):
            S3Tool().run()

    def test_s3_get_requires_key(self):
        from draf.tool.builtin import S3GetTool

        with pytest.raises(ValueError, match="key"):
            S3GetTool({"bucket": "b"}).run()

    def test_s3_put_requires_key(self):
        from draf.tool.builtin import S3PutTool

        with pytest.raises(ValueError, match="key"):
            S3PutTool({"bucket": "b"}).run()

    def test_s3_missing_boto3(self, monkeypatch):
        import builtins

        from draf.tool.builtin import S3Tool

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("no boto3")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="boto3"):
            S3Tool({"bucket": "b"}).run()

    def test_slack_requires_token(self):
        from draf.tool.builtin import SlackSendTool

        with pytest.raises(ValueError, match="token"):
            SlackSendTool().run(text="hi")

    def test_slack_requires_channel(self):
        from draf.tool.builtin import SlackSendTool

        with pytest.raises(ValueError, match="channel"):
            SlackSendTool({"token": "x"}).run(text="hi")

    def test_slack_missing_sdk(self, monkeypatch):
        import builtins

        from draf.tool.builtin import SlackSendTool

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "slack_sdk":
                raise ImportError("no slack-sdk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="slack-sdk"):
            SlackSendTool({"token": "x", "channel": "#c"}).run(text="hi")

    def test_sql_query_sqlite_select(self, tmp_path):
        import sqlite3

        from draf.tool.builtin import SQLQueryTool

        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.executemany(
            "INSERT INTO users VALUES (?, ?)",
            [(1, "alice"), (2, "bob"), (3, "carol")],
        )
        conn.commit()
        conn.close()

        tool = SQLQueryTool({"db_type": "sqlite", "path": str(db)})
        result = tool.run(query="SELECT id, name FROM users WHERE id > 1 ORDER BY id")
        assert "bob" in result
        assert "carol" in result
        assert "alice" not in result

    def test_sql_query_sqlite_requires_path(self):
        from draf.tool.builtin import SQLQueryTool

        with pytest.raises(ValueError, match="path"):
            SQLQueryTool({"db_type": "sqlite"}).run(query="SELECT 1")

    def test_sql_query_rejects_writes(self, tmp_path):
        from draf.tool.builtin import SQLQueryTool

        with pytest.raises(ValueError, match="read-only"):
            SQLQueryTool({"db_type": "sqlite", "path": str(tmp_path / "x.db")}).run(
                query="INSERT INTO users VALUES (1, 'x')"
            )

    def test_sql_query_requires_query(self):
        from draf.tool.builtin import SQLQueryTool

        with pytest.raises(ValueError, match="query"):
            SQLQueryTool().run()

    def test_sql_query_postgres_missing_psycopg(self, monkeypatch):
        import builtins

        from draf.tool.builtin import SQLQueryTool

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("no psycopg")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="psycopg"):
            SQLQueryTool({"db_type": "postgres", "dsn": "postgresql://x"}).run(
                query="SELECT 1"
            )

    def test_registered_in_default_registry(self):
        from draf.tool.registry import default_tool_registry

        for name in (
            "fetch_url",
            "read_pdf",
            "s3_list",
            "s3_get",
            "s3_put",
            "slack_send",
            "sql_query",
            "sql_list_tables",
            "sql_describe",
            "list_dir",
            "glob",
            "getenv",
            "current_time",
            "json_parse",
            "yaml_parse",
            "kv_store",
            "python_eval",
            "http_request",
            "send_email",
            "send_telegram",
        ):
            assert name in default_tool_registry.list(), f"{name} not registered"


class TestFsEnvTools:
    def test_list_dir(self, tmp_path):
        from draf.tool.builtin import ListDirTool

        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "sub").mkdir()
        result = ListDirTool().run(path=str(tmp_path))
        assert "a.txt" in result
        assert "sub" in result

    def test_list_dir_recursive(self, tmp_path):
        from draf.tool.builtin import ListDirTool

        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("x")
        result = ListDirTool().run(path=str(tmp_path), recursive=True)
        assert "b.txt" in result

    def test_list_dir_not_a_directory(self, tmp_path):
        from draf.tool.builtin import ListDirTool

        with pytest.raises(ValueError, match="not a directory"):
            ListDirTool().run(path=str(tmp_path / "nope"))

    def test_glob(self, tmp_path):
        from draf.tool.builtin import GlobTool

        (tmp_path / "one.txt").write_text("x")
        (tmp_path / "two.md").write_text("x")
        result = GlobTool().run(pattern=str(tmp_path / "*.txt"))
        assert "one.txt" in result
        assert "two.md" not in result

    def test_glob_requires_pattern(self):
        from draf.tool.builtin import GlobTool

        with pytest.raises(ValueError, match="pattern"):
            GlobTool().run()

    def test_getenv(self, monkeypatch):
        from draf.tool.builtin import GetEnvTool

        monkeypatch.setenv("DRAF_TEST_VAR", "hello")
        assert GetEnvTool().run(name="DRAF_TEST_VAR") == "hello"
        assert GetEnvTool().run(name="DRAF_MISSING") == "not set"

    def test_getenv_masks_secrets(self, monkeypatch):
        from draf.tool.builtin import GetEnvTool

        monkeypatch.setenv("MY_API_KEY", "supersecret")
        assert GetEnvTool().run(name="MY_API_KEY") == "***"
        assert (
            GetEnvTool({"mask_secrets": False}).run(name="MY_API_KEY") == "supersecret"
        )

    def test_getenv_requires_name(self):
        from draf.tool.builtin import GetEnvTool

        with pytest.raises(ValueError, match="name"):
            GetEnvTool().run()

    def test_current_time(self):
        from draf.tool.builtin import CurrentTimeTool

        result = CurrentTimeTool().run()
        assert "T" in result

    def test_current_time_unknown_tz(self):
        from draf.tool.builtin import CurrentTimeTool

        with pytest.raises(ValueError, match="timezone"):
            CurrentTimeTool().run(timezone="Not/AZone")


class TestDataTools:
    def test_json_parse(self):
        from draf.tool.builtin import JsonParseTool

        assert JsonParseTool().run(text='{"a": 1}') == '{\n  "a": 1\n}'

    def test_json_parse_invalid(self):
        from draf.tool.builtin import JsonParseTool

        with pytest.raises(ValueError, match="invalid JSON"):
            JsonParseTool().run(text="{bad}")

    def test_yaml_parse(self):
        from draf.tool.builtin import YamlParseTool

        result = YamlParseTool().run(text="a: 1\nb:\n  - x\n  - y")
        assert '"a": 1' in result
        assert '"x"' in result

    def test_kv_store_roundtrip(self, tmp_path):
        from draf.tool.builtin import KVStoreTool

        path = str(tmp_path / "kv.json")
        tool = KVStoreTool({"path": path})
        assert tool.run(action="set", key="name", value='"alice"') == "set name"
        assert tool.run(action="get", key="name") == '"alice"'
        assert "name" in tool.run(action="list")
        assert tool.run(action="delete", key="name") == "deleted name"
        assert tool.run(action="get", key="name") == "not found"

    def test_kv_store_unknown_action(self, tmp_path):
        from draf.tool.builtin import KVStoreTool

        with pytest.raises(ValueError, match="action"):
            KVStoreTool({"path": str(tmp_path / "kv.json")}).run(action="bogus")

    def test_python_eval_arithmetic(self):
        from draf.tool.builtin import PythonEvalTool

        tool = PythonEvalTool()
        assert tool.run(expression="2 + 3 * 4") == "14"
        assert tool.run(expression="10 / 2") == "5.0"
        assert tool.run(expression="len([1, 2, 3])") == "3"
        assert tool.run(expression="math.sqrt(16)") == "4.0"
        assert tool.run(expression="2 ** 10") == "1024"

    def test_python_eval_rejects_unsafe(self):
        from draf.tool.builtin import PythonEvalTool

        tool = PythonEvalTool()
        with pytest.raises(ValueError, match="not allowed"):
            tool.run(expression="__import__('os')")
        with pytest.raises(ValueError, match="not allowed"):
            tool.run(expression="os.system('ls')")


class TestHttpTool:
    @pytest.mark.asyncio
    async def test_http_request(self, monkeypatch):
        import httpx

        from draf.tool.builtin import HttpRequestTool

        class FakeResponse:
            status_code = 200
            text = "hello"

            @property
            def headers(self):
                return {"Content-Type": "text/plain"}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def request(self, *a, **k):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())
        result = await HttpRequestTool().arun(url="http://x")
        assert "200" in result
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_http_request_requires_url(self):
        from draf.tool.builtin import HttpRequestTool

        with pytest.raises(ValueError, match="url"):
            await HttpRequestTool().arun()


class TestSQLSchemaTools:
    def test_sql_list_tables(self, tmp_path):
        import sqlite3

        from draf.tool.builtin import SQLListTablesTool

        db = tmp_path / "db.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("CREATE TABLE orders (id INTEGER)")
        conn.commit()
        conn.close()

        result = SQLListTablesTool({"db_type": "sqlite", "path": str(db)}).run()
        assert "users" in result
        assert "orders" in result

    def test_sql_describe(self, tmp_path):
        import sqlite3

        from draf.tool.builtin import SQLDescribeTool

        db = tmp_path / "db.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
        conn.close()

        result = SQLDescribeTool({"db_type": "sqlite", "path": str(db)}).run(
            table="users"
        )
        assert "id" in result
        assert "name" in result

    def test_sql_describe_requires_table(self):
        from draf.tool.builtin import SQLDescribeTool

        with pytest.raises(ValueError, match="table"):
            SQLDescribeTool().run()


class TestNotifyTools:
    def test_send_email_requires_host(self):
        from draf.tool.builtin import SendEmailTool

        with pytest.raises(ValueError, match="host"):
            SendEmailTool().run(to="a@b.c", subject="hi", body="x")

    def test_send_email_requires_from(self):
        from draf.tool.builtin import SendEmailTool

        with pytest.raises(ValueError, match="from_addr"):
            SendEmailTool({"host": "smtp.example.com"}).run(
                to="a@b.c", subject="hi", body="x"
            )

    @pytest.mark.asyncio
    async def test_send_telegram_requires_token(self):
        from draf.tool.builtin import SendTelegramTool

        with pytest.raises(ValueError, match="token"):
            await SendTelegramTool().arun(text="hi")

    @pytest.mark.asyncio
    async def test_send_telegram_requires_chat_id(self):
        from draf.tool.builtin import SendTelegramTool

        with pytest.raises(ValueError, match="chat_id"):
            await SendTelegramTool({"token": "x"}).arun(text="hi")

    @pytest.mark.asyncio
    async def test_send_telegram(self, monkeypatch):
        import httpx

        from draf.tool.builtin import SendTelegramTool

        class FakeResponse:
            def raise_for_status(self):
                pass

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())
        result = await SendTelegramTool({"token": "x", "chat_id": "123"}).arun(
            text="hello"
        )
        assert "123" in result


class TestShellSandbox:
    @pytest.mark.asyncio
    async def test_blocked_commands(self):
        from draf.tool.builtin import ShellTool

        t = ShellTool()
        with pytest.raises(PermissionError, match="blocked"):
            await t.arun("dd if=/dev/zero of=/tmp/x bs=1 count=1")
        with pytest.raises(PermissionError, match="blocked"):
            await t.arun("sudo ls")

    @pytest.mark.asyncio
    async def test_allowed_commands_whitelist(self):
        from draf.tool.builtin import ShellTool

        t = ShellTool(allowed_commands=["echo"])
        with pytest.raises(PermissionError, match="not allowed"):
            await t.arun("ls /tmp")
        r = await t.arun("echo hello")
        assert r == "hello"

    @pytest.mark.asyncio
    async def test_allowed_commands_overrides_blocked(self):
        from draf.tool.builtin import ShellTool

        t = ShellTool(allowed_commands=["dd"])
        with pytest.raises(PermissionError, match="blocked"):
            await t.arun("dd if=/dev/zero of=/tmp/x bs=1 count=1")

    @pytest.mark.asyncio
    async def test_empty_command_raises(self):
        from draf.tool.builtin import ShellTool

        t = ShellTool()
        with pytest.raises(ValueError, match="empty"):
            await t.arun("")

    @pytest.mark.asyncio
    async def test_shell_metacharacters_rejected(self):
        """Blocklist bypasses via &&/;/|/backticks/$(...) are impossible:
        the tool runs execve, and tokens with shell metacharacters are
        refused outright."""
        from draf.tool.builtin import ShellTool

        t = ShellTool()
        for cmd in (
            "echo hi && sudo rm -rf /",
            "echo hi; reboot",
            "echo hi | dd of=/dev/null",
            "echo $(sudo id)",
            "echo `sudo id`",
            "ls *.py",
            "cat /tmp/f > /etc/passwd",
        ):
            with pytest.raises(PermissionError, match="metacharacters"):
                await t.arun(cmd)
