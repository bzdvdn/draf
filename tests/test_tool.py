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
        from draf.tool import Tool
        import asyncio

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
        from draf.tool import default_tool_registry, tool
        import asyncio

        @tool("async_hello")
        async def greet(name: str = "") -> str:
            return f"Hi {name}"

        t = default_tool_registry.create("async_hello")
        r = asyncio.run(t.arun(name="Test"))
        assert r == "Hi Test"


class TestToolRegistry:
    def test_register_and_list(self):
        from draf.tool import ToolRegistry, Tool

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
        from draf.tool.builtin.file import WriteFileTool, ReadFileTool, EditFileTool

        path = str(tmp_path / "test.txt")
        wt = WriteFileTool()
        wt.run(path=path, content="hello world")
        rt = ReadFileTool()
        assert rt.run(path=path) == "hello world"
        et = EditFileTool()
        et.run(path=path, old="world", new="there")
        assert rt.run(path=path) == "hello there"


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
