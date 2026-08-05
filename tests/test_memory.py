import asyncio
import math

import pytest


class _TokenEmbedder:
    """Deterministic bag-of-tokens embedder (16 dims, normalised)."""

    dim = 16

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            vec[sum(ord(c) for c in tok) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture
def store():
    from draf.rag.stores import InMemoryVectorStore

    return InMemoryVectorStore(dim=_TokenEmbedder.dim)


@pytest.fixture
def embedder():
    return _TokenEmbedder()


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_put_get_round_trip(self, store, embedder):
        from draf.memory import MemoryStore

        mem = MemoryStore(store, embedder)
        await mem.put(("users", "u1"), "prefs", {"text": "prefers email over Slack"})
        item = await mem.get(("users", "u1"), "prefs")
        assert item is not None
        assert item.value["text"] == "prefers email over Slack"
        assert item.namespace == ("users", "u1")
        assert item.key == "prefs"

    @pytest.mark.asyncio
    async def test_put_is_upsert(self, store, embedder):
        from draf.memory import MemoryStore

        mem = MemoryStore(store, embedder)
        await mem.put(("users", "u1"), "prefs", {"text": "old"})
        await mem.put(("users", "u1"), "prefs", {"text": "new", "channel": "slack"})
        item = await mem.get(("users", "u1"), "prefs")
        assert item.value["text"] == "new"
        assert item.value["channel"] == "slack"

    @pytest.mark.asyncio
    async def test_put_requires_text(self, store, embedder):
        from draf.memory import MemoryStore

        mem = MemoryStore(store, embedder)
        with pytest.raises(ValueError, match="'text'"):
            await mem.put(("u",), "k", {"other": 1})

    @pytest.mark.asyncio
    async def test_namespace_subtree_isolation(self, store, embedder):
        from draf.memory import MemoryStore

        mem = MemoryStore(store, embedder)
        await mem.put(("users", "u1", "memories"), "a", {"text": "pizza"})
        await mem.put(("users", "u2", "memories"), "b", {"text": "sushi"})
        hits = await mem.search(("users", "u1"), query="food")
        assert [i.key for i in hits] == ["a"]
        hits = await mem.search(("users",), query="food")
        assert {i.key for i in hits} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_search_ranks_by_similarity(self, store, embedder):
        from draf.memory import MemoryStore

        mem = MemoryStore(store, embedder)
        await mem.put(("u",), "contact", {"text": "email her at work"})
        await mem.put(("u",), "meal", {"text": "likes sushi on fridays"})
        hits = await mem.search(("u",), query="how to reach her", k=2)
        assert hits[0].key == "contact"
        assert hits[0].score is not None

    @pytest.mark.asyncio
    async def test_search_without_query_is_recency(self, store, embedder):
        from draf.memory import MemoryStore

        mem = MemoryStore(store, embedder)
        await mem.put(("u",), "first", {"text": "old fact"})
        await asyncio.sleep(0.01)
        await mem.put(("u",), "second", {"text": "new fact"})
        hits = await mem.search(("u",))
        assert hits[0].key == "second"
        assert hits[0].score is None

    @pytest.mark.asyncio
    async def test_ttl_expires_items(self, store, embedder):
        from draf.memory import MemoryStore

        mem = MemoryStore(store, embedder, ttl=0.05)
        await mem.put(("u",), "short", {"text": "ephemeral"})
        assert await mem.get(("u",), "short") is not None
        await asyncio.sleep(0.08)
        assert await mem.get(("u",), "short") is None
        assert await mem.search(("u",)) == []

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired(self, store, embedder):
        from draf.memory import MemoryStore

        mem = MemoryStore(store, embedder, ttl=0.05)
        await mem.put(("u",), "a", {"text": "ephemeral"})
        await mem.put(("u",), "b", {"text": "durable"}, ttl=None)
        await asyncio.sleep(0.08)
        assert await mem.cleanup() == 1
        assert [i.key for i in await mem.search(("u",))] == ["b"]

    @pytest.mark.asyncio
    async def test_list_and_delete(self, store, embedder):
        from draf.memory import MemoryStore

        mem = MemoryStore(store, embedder)
        await mem.put(("u",), "a", {"text": "one"})
        await mem.put(("u",), "b", {"text": "two"})
        assert set(await mem.list(("u",))) == {"a", "b"}
        await mem.delete(("u",), "a")
        assert await mem.get(("u",), "a") is None
        assert await mem.get(("u",), "b") is not None


class TestMemoryTool:
    @pytest.mark.asyncio
    async def test_remember_recall_cycle(self, store, embedder):
        from draf.memory import MemoryTool

        tool = MemoryTool(store=store, embedder=embedder, namespace=("users", "u1"))
        out = await tool.arun(
            action="remember",
            text="prefers email over Slack",
            metadata={"channel": "email"},
        )
        assert out.startswith("remembered")
        out = await tool.arun(action="recall", query="how to reach them")
        assert "email" in out

    @pytest.mark.asyncio
    async def test_forget_and_list(self, store, embedder):
        from draf.memory import MemoryTool

        tool = MemoryTool(store=store, embedder=embedder)
        await tool.arun(action="remember", namespace=["u"], key="a", text="one")
        await tool.arun(action="remember", namespace=["u"], key="b", text="two")
        out = await tool.arun(action="list", namespace=["u"])
        assert "a" in out and "b" in out
        await tool.arun(action="forget", namespace=["u"], key="a")
        out = await tool.arun(action="list", namespace=["u"])
        assert "a" not in out and "b" in out

    @pytest.mark.asyncio
    async def test_dedup_overwrites_similar(self, store, embedder):
        from draf.memory import MemoryTool

        tool = MemoryTool(
            store=store, embedder=embedder, namespace=("u",), similarity_threshold=0.8
        )
        await tool.arun(action="remember", key="first", text="email her at work")
        await tool.arun(
            action="remember", key="second", text="email her at work please"
        )
        keys = await tool.memory.list(("u",))
        assert len(keys) == 1
        assert keys[0] == "first"

    @pytest.mark.asyncio
    async def test_unknown_action_raises(self, store, embedder):
        from draf.memory import MemoryTool

        tool = MemoryTool(store=store, embedder=embedder)
        with pytest.raises(ValueError, match="unknown memory action"):
            await tool.arun(action="bogus")

    def test_builtin_registry_has_memory_tool(self):
        import draf.tool.builtin  # noqa: F401 — registers built-in tools
        from draf.tool.registry import default_tool_registry

        assert "memory" in default_tool_registry.list()
        cls = default_tool_registry.create("memory")
        assert cls.name == "memory"
