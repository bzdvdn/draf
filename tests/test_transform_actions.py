import pytest

from teff.node import Transform


class TestContains:
    @pytest.mark.asyncio
    async def test_true_when_substring(self):
        node = Transform(
            action="contains", input_key="text", value="refund", output_key="has_refund"
        )
        out = await node.execute(None, {"text": "the user asked for a refund"})
        assert out == {"has_refund": "true"}

    @pytest.mark.asyncio
    async def test_false_when_missing(self):
        node = Transform(
            action="contains", input_key="text", value="refund", output_key="has_refund"
        )
        out = await node.execute(None, {"text": "the user is happy"})
        assert out == {"has_refund": "false"}


class TestCompare:
    @pytest.mark.asyncio
    async def test_numeric_ge(self):
        node = Transform(
            action="compare", input_key="score", value="0.8", op="ge", output_key="ok"
        )
        out = await node.execute(None, {"score": "0.92"})
        assert out == {"ok": "true"}

    @pytest.mark.asyncio
    async def test_numeric_lt(self):
        node = Transform(
            action="compare", input_key="score", value="0.8", op="lt", output_key="ok"
        )
        out = await node.execute(None, {"score": "0.5"})
        assert out == {"ok": "true"}

    @pytest.mark.asyncio
    async def test_string_eq(self):
        node = Transform(
            action="compare", input_key="who", value="ADMIN_OK", output_key="ok"
        )
        out = await node.execute(None, {"who": "ADMIN_OK"})
        assert out == {"ok": "true"}

    @pytest.mark.asyncio
    async def test_ne(self):
        node = Transform(
            action="compare",
            input_key="who",
            value="ADMIN_OK",
            op="ne",
            output_key="ok",
        )
        out = await node.execute(None, {"who": "GUEST"})
        assert out == {"ok": "true"}

    @pytest.mark.asyncio
    async def test_unknown_op_raises(self):
        node = Transform(
            action="compare", input_key="x", value="1", op="bogus", output_key="ok"
        )
        with pytest.raises(ValueError, match="bogus"):
            await node.execute(None, {"x": "1"})


class TestSplitJoin:
    @pytest.mark.asyncio
    async def test_split_default_sep(self):
        node = Transform(action="split", input_key="csv", output_key="items")
        out = await node.execute(None, {"csv": "a,b,c"})
        assert out == {"items": ["a", "b", "c"]}

    @pytest.mark.asyncio
    async def test_split_custom_sep(self):
        node = Transform(action="split", input_key="line", sep=" ", output_key="words")
        out = await node.execute(None, {"line": "one two three"})
        assert out == {"words": ["one", "two", "three"]}

    @pytest.mark.asyncio
    async def test_join_default_sep(self):
        node = Transform(action="join", input_key="items", output_key="csv")
        out = await node.execute(None, {"items": ["a", "b", "c"]})
        assert out == {"csv": "a,b,c"}

    @pytest.mark.asyncio
    async def test_join_custom_sep(self):
        node = Transform(action="join", input_key="items", sep=" | ", output_key="line")
        out = await node.execute(None, {"items": ["x", "y"]})
        assert out == {"line": "x | y"}


class TestReplace:
    @pytest.mark.asyncio
    async def test_replaces_old_with_new(self):
        node = Transform(
            action="replace", input_key="text", old="foo", new="bar", output_key="out"
        )
        out = await node.execute(None, {"text": "foo-foo"})
        assert out == {"out": "bar-bar"}

    @pytest.mark.asyncio
    async def test_removes_when_no_new(self):
        node = Transform(action="replace", input_key="text", old="-", output_key="out")
        out = await node.execute(None, {"text": "a-b-c"})
        assert out == {"out": "abc"}


class TestCoalesce:
    @pytest.mark.asyncio
    async def test_uses_input_when_set(self):
        node = Transform(
            action="coalesce", input_key="title", value="Untitled", output_key="out"
        )
        out = await node.execute(None, {"title": "The Report"})
        assert out == {"out": "The Report"}

    @pytest.mark.asyncio
    async def test_falls_back_when_empty(self):
        node = Transform(
            action="coalesce", input_key="title", value="Untitled", output_key="out"
        )
        out = await node.execute(None, {"title": ""})
        assert out == {"out": "Untitled"}


class TestPick:
    @pytest.mark.asyncio
    async def test_extracts_field(self):
        node = Transform(action="pick", input_key="obj", field="name", output_key="n")
        out = await node.execute(None, {"obj": {"name": "Ada", "id": 7}})
        assert out == {"n": "Ada"}

    @pytest.mark.asyncio
    async def test_raw_keeps_value(self):
        node = Transform(
            action="pick", input_key="obj", field="tags", output_key="t", raw=True
        )
        out = await node.execute(None, {"obj": {"tags": ["a"]}})
        assert out == {"t": ["a"]}


class TestNumberCoercion:
    @pytest.mark.asyncio
    async def test_to_int(self):
        node = Transform(action="to_int", input_key="n", output_key="out")
        out = await node.execute(None, {"n": "42.0"})
        assert out == {"out": "42"}

    @pytest.mark.asyncio
    async def test_to_float(self):
        node = Transform(action="to_float", input_key="n", output_key="out")
        out = await node.execute(None, {"n": "42"})
        assert out == {"out": "42.0"}


class TestNow:
    @pytest.mark.asyncio
    async def test_writes_utc_iso_timestamp(self):
        node = Transform(action="now", output_key="ts")
        out = await node.execute(None, {})
        assert out["ts"].endswith("+00:00")
