import math

import pytest

from draf.harness.loop import ModelReply


class _FakeHarness:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[list[dict]] = []

    async def call(self, messages, **kwargs):
        self.calls.append(messages)
        return ModelReply(
            data={},
            message={"role": "assistant", "content": self.content},
            content=self.content,
        )


class _TokenEmbedder:
    dim = 16

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            vec[sum(ord(c) for c in tok) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class TestMemoryExtractor:
    @pytest.mark.asyncio
    async def test_extract_plain_json(self):
        from draf.memory import MemoryExtractor

        harness = _FakeHarness(
            '[{"text": "user prefers email"}, {"text": "user likes sushi"}]'
        )
        facts = await MemoryExtractor(harness).extract([])
        assert facts == ["user prefers email", "user likes sushi"]
        assert harness.calls[0][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_extract_tolerates_fence_and_prose(self):
        from draf.memory import MemoryExtractor

        harness = _FakeHarness(
            'Here you go:\n```json\n[{"text": "works remotely"}]\n```\nDone.'
        )
        facts = await MemoryExtractor(harness).extract([])
        assert facts == ["works remotely"]

    @pytest.mark.asyncio
    async def test_extract_handles_string_items_and_empty(self):
        from draf.memory import MemoryExtractor

        h1 = _FakeHarness('["prefers dark mode"]')
        assert await MemoryExtractor(h1).extract([]) == ["prefers dark mode"]

        h2 = _FakeHarness("[]")
        assert await MemoryExtractor(h2).extract([]) == []

    @pytest.mark.asyncio
    async def test_extract_invalid_json_returns_empty(self):
        from draf.memory import MemoryExtractor

        harness = _FakeHarness("I could not find any facts.")
        assert await MemoryExtractor(harness).extract([]) == []

    @pytest.mark.asyncio
    async def test_extract_builds_transcript(self):
        from draf.memory import MemoryExtractor

        harness = _FakeHarness("[]")
        await MemoryExtractor(harness).extract(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        transcript = harness.calls[0][1]["content"]
        assert "user: hi" in transcript
        assert "assistant: hello" in transcript

    @pytest.mark.asyncio
    async def test_requires_model_or_harness(self):
        from draf.memory import MemoryExtractor

        with pytest.raises(ValueError, match="harness|model"):
            MemoryExtractor()

    @pytest.mark.asyncio
    async def test_save_writes_stable_facts(self):
        from draf.memory import MemoryExtractor, MemoryStore
        from draf.rag.stores import InMemoryVectorStore

        embedder = _TokenEmbedder()
        mem = MemoryStore(InMemoryVectorStore(dim=16), embedder)
        harness = _FakeHarness('[{"text": "loves hiking"}]')
        written = await MemoryExtractor(harness).save(mem, [], ("u",))
        assert len(written) == 1
        key, fact = written[0]
        assert fact == "loves hiking"
        hit = await mem.get(("u",), key)
        assert hit is not None
        assert hit.value["text"] == "loves hiking"
        assert hit.value["source"] == "extractor"


class TestMemoryContext:
    @pytest.mark.asyncio
    async def test_memory_context_formats_block(self):
        from draf.memory import MemoryStore, memory_context
        from draf.rag.stores import InMemoryVectorStore

        embedder = _TokenEmbedder()
        mem = MemoryStore(InMemoryVectorStore(dim=16), embedder)
        await mem.put(("u",), "a", {"text": "prefers email"})
        await mem.put(("u",), "b", {"text": "likes sushi"})
        block = await memory_context(mem, "email", namespace=("u",), k=5)
        assert block.startswith("Relevant memories:")
        assert "prefers email" in block
        assert "- " in block

    @pytest.mark.asyncio
    async def test_memory_context_empty_without_match(self):
        from draf.memory import MemoryStore, memory_context
        from draf.rag.stores import InMemoryVectorStore

        mem = MemoryStore(InMemoryVectorStore(dim=16), _TokenEmbedder())
        block = await memory_context(mem, "anything")
        assert block == ""

    @pytest.mark.asyncio
    async def test_memory_context_blank_query_is_empty(self):
        from draf.memory import memory_context

        class _Null:
            async def search(self, *a, **k):
                raise AssertionError("should not search")

        assert await memory_context(_Null(), "   ") == ""

    def test_last_user_text(self):
        from draf.memory import last_user_text

        msgs = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "  hello  "},
        ]
        assert last_user_text(msgs) == "hello"
        assert last_user_text([], fallback="fb") == "fb"


class TestMemoryContextFromConfig:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        from draf.memory import memory_context_from_config

        assert await memory_context_from_config({}, state={}, ctx=None) == ""

    @pytest.mark.asyncio
    async def test_object_store_recalls_block(self):
        from draf.memory import MemoryStore, memory_context_from_config
        from draf.rag.stores import InMemoryVectorStore

        embedder = _TokenEmbedder()
        mem = MemoryStore(InMemoryVectorStore(dim=16), embedder)
        await mem.put(("users", "ana"), "prefs", {"text": "prefers email"})
        state = {
            "input": "how to reach ana?",
            "messages": [{"role": "user", "content": "how to reach ana?"}],
        }
        block = await memory_context_from_config(
            {"memory": {"store": mem, "namespace": ("users", "ana"), "k": 5}},
            state=state,
            ctx=None,
        )
        assert "prefers email" in block

    @pytest.mark.asyncio
    async def test_dict_store_without_embeddings_is_empty(self):
        from draf.memory import memory_context_from_config

        state = {}
        block = await memory_context_from_config(
            {
                "memory": {
                    "store": {"type": "in_memory", "dim": 16},
                    "embedder": {
                        "provider": "ollama",
                        "model": "nomic-embed-text",
                    },
                    "namespace": "u",
                }
            },
            state=state,
            ctx=None,
        )
        assert block == ""

    @pytest.mark.asyncio
    async def test_memory_config_object(self):
        from draf.memory import MemoryConfig, MemoryStore, memory_context_from_config
        from draf.rag.stores import InMemoryVectorStore

        embedder = _TokenEmbedder()
        mem = MemoryStore(InMemoryVectorStore(dim=16), embedder)
        await mem.put(("u",), "a", {"text": "loves hiking"})
        state = {"input": "what about sports?", "messages": []}
        block = await memory_context_from_config(
            {"memory": MemoryConfig(store=mem, namespace=("u",), k=5)},
            state=state,
            ctx=None,
        )
        assert "loves hiking" in block

    def test_memory_config_to_dict(self):
        from draf.memory import MemoryConfig

        c = MemoryConfig(store="x", namespace=("u",), k=3, header="Mem:")
        assert c.to_dict() == {
            "store": "x",
            "namespace": ("u",),
            "k": 3,
            "header": "Mem:",
        }


class TestMemoryFromConfig:
    @pytest.mark.asyncio
    async def test_builds_in_memory_store(self):
        from draf.memory.tool import memory_from_config

        mem = memory_from_config({})
        assert mem is not None
        assert await mem.search(()) == []

    @pytest.mark.asyncio
    async def test_uses_ttl_default(self):
        from draf.memory.tool import memory_from_config

        mem = memory_from_config({"ttl": 60})
        assert mem._ttl == 60
