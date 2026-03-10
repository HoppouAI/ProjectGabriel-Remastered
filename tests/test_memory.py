"""
Tests for the ProjectGabriel memory system.
Covers: MemorySystem CRUD, filtering, prompt injection, recall sub-agent, and tool handler.
"""
import asyncio
import os
import sys
import time

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import (
    MemorySystem,
    MEMORY_TYPE_LONG_TERM,
    MEMORY_TYPE_SHORT_TERM,
    MEMORY_TYPE_QUICK_NOTE,
    _hash_content,
    format_memories_for_prompt,
    get_memory_content_for_prompt,
    memory_system,
    recall_memories,
    handle_memory_function_call,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem():
    """Create a fresh in-memory SQLite MemorySystem for each test."""
    ms = MemorySystem.__new__(MemorySystem)
    ms.config = {}
    ms.backend = "sqlite"
    ms.mongo_uri = ""
    ms.mongo_db = ""
    ms.mongo_collection_name = ""
    ms.sqlite_path = ":memory:"
    ms.quick_note_ttl_hours = 6
    ms.short_term_ttl_days = 7
    ms.note_min_interval = 0.1
    ms.dedupe_window = 5
    ms.client = None
    ms.collection = None
    ms.sqlite_conn = None
    ms._sqlite_lock = __import__("threading").RLock()
    ms._cleanup_running = False
    ms._cleanup_thread = None
    ms._note_last_ts = 0
    ms._note_last_hash = ""

    import sqlite3
    ms.sqlite_conn = sqlite3.connect(":memory:", check_same_thread=False)
    ms.sqlite_conn.row_factory = sqlite3.Row
    ms.sqlite_conn.execute("PRAGMA journal_mode=WAL")
    ms._init_sqlite_tables()
    return ms


# ---------------------------------------------------------------------------
# Basic CRUD Tests
# ---------------------------------------------------------------------------

class TestMemoryCRUD:
    def test_save_and_read(self, mem):
        res = mem.save("test_key", "hello world", category="test")
        assert res["success"]
        assert res["key"] == "test_key"

        read = mem.read("test_key")
        assert read["success"]
        assert read["memory"]["content"] == "hello world"
        assert read["memory"]["category"] == "test"

    def test_save_upsert(self, mem):
        mem.save("key1", "version 1")
        mem.save("key1", "version 2")
        read = mem.read("key1")
        assert read["memory"]["content"] == "version 2"

    def test_read_nonexistent(self, mem):
        res = mem.read("nonexistent")
        assert not res["success"]

    def test_update(self, mem):
        mem.save("upd", "original content")
        res = mem.update("upd", content="updated content", category="new_cat")
        assert res["success"]

        read = mem.read("upd")
        assert read["memory"]["content"] == "updated content"
        assert read["memory"]["category"] == "new_cat"

    def test_update_nonexistent(self, mem):
        res = mem.update("nope", content="x")
        assert not res["success"]

    def test_delete(self, mem):
        mem.save("del_me", "temp")
        res = mem.delete("del_me")
        assert res["success"]

        read = mem.read("del_me")
        assert not read["success"]

    def test_delete_nonexistent(self, mem):
        res = mem.delete("nope")
        assert not res["success"]


# ---------------------------------------------------------------------------
# List & Search Tests
# ---------------------------------------------------------------------------

class TestListAndSearch:
    def test_list_all(self, mem):
        mem.save("a", "apple", memory_type=MEMORY_TYPE_LONG_TERM)
        mem.save("b", "banana", memory_type=MEMORY_TYPE_SHORT_TERM)
        mem.save("c", "cherry", memory_type=MEMORY_TYPE_QUICK_NOTE)

        res = mem.list_memories()
        assert res["success"]
        assert res["count"] == 3

    def test_list_by_type(self, mem):
        mem.save("lt1", "long term 1", memory_type=MEMORY_TYPE_LONG_TERM)
        mem.save("lt2", "long term 2", memory_type=MEMORY_TYPE_LONG_TERM)
        mem.save("st1", "short term 1", memory_type=MEMORY_TYPE_SHORT_TERM)

        # This was previously bugged — should now correctly filter long_term
        res = mem.list_memories(memory_type=MEMORY_TYPE_LONG_TERM)
        assert res["success"]
        assert res["count"] == 2

    def test_list_by_category(self, mem):
        mem.save("p1", "person info", category="personal")
        mem.save("w1", "work info", category="work")
        mem.save("p2", "more personal", category="personal")

        res = mem.list_memories(category="personal")
        assert res["success"]
        assert res["count"] == 2

    def test_search(self, mem):
        mem.save("fruit1", "I love apples and oranges")
        mem.save("fruit2", "Bananas are great")
        mem.save("other", "The weather is nice")

        res = mem.search("apple")
        assert res["success"]
        assert res["count"] >= 1
        assert any("apple" in m["content"].lower() for m in res["memories"])

    def test_search_by_key(self, mem):
        mem.save("user_alice", "Alice is 25 years old")
        mem.save("user_bob", "Bob likes cats")

        res = mem.search("alice")
        assert res["success"]
        assert res["count"] >= 1

    def test_search_with_type_filter(self, mem):
        mem.save("slt", "long term search target", memory_type=MEMORY_TYPE_LONG_TERM)
        mem.save("sst", "short term search target", memory_type=MEMORY_TYPE_SHORT_TERM)

        res = mem.search("target", memory_type=MEMORY_TYPE_LONG_TERM)
        assert res["success"]
        assert res["count"] == 1
        assert res["memories"][0]["key"] == "slt"


# ---------------------------------------------------------------------------
# Stats & Deduplication Tests
# ---------------------------------------------------------------------------

class TestStatsAndDedupe:
    def test_stats(self, mem):
        mem.save("a", "x", memory_type=MEMORY_TYPE_LONG_TERM)
        mem.save("b", "y", memory_type=MEMORY_TYPE_SHORT_TERM)
        mem.save("c", "z", memory_type=MEMORY_TYPE_QUICK_NOTE)

        res = mem.stats()
        assert res["success"]
        assert res["stats"]["total"] == 3
        assert res["stats"]["long_term"] == 1
        assert res["stats"]["short_term"] == 1
        assert res["stats"]["quick_note"] == 1

    def test_deduplication(self, mem):
        content = "This is the same content"
        h = _hash_content(content)

        mem.save("dup1", content, memory_type=MEMORY_TYPE_QUICK_NOTE)
        assert mem.has_recent_duplicate(h, 60, [MEMORY_TYPE_QUICK_NOTE])

    def test_no_false_duplicate(self, mem):
        h = _hash_content("something unique")
        assert not mem.has_recent_duplicate(h, 60)

    def test_access_count_increments(self, mem):
        mem.save("accessed", "some content")

        mem.read("accessed")
        mem.read("accessed")
        mem.read("accessed")

        res = mem.read("accessed")
        assert res["memory"]["access_count"] == 4  # 4th read


# ---------------------------------------------------------------------------
# Prompt Injection Tests
# ---------------------------------------------------------------------------

class TestPromptInjection:
    def test_format_memories_for_prompt(self):
        memories = [
            {"key": "k1", "content": "Alice is 25", "category": "personal"},
            {"key": "k2", "content": "Bob likes cats", "category": "facts"},
        ]
        result = format_memories_for_prompt(memories)
        assert "Alice is 25" in result
        assert "Bob likes cats" in result
        assert "[k1]" in result

    def test_format_truncation(self):
        memories = [{"key": "long", "content": "x" * 500, "category": "test"}]
        result = format_memories_for_prompt(memories, max_length=50)
        assert "..." in result
        assert len(result) < 200  # Much shorter than 500

    def test_get_recent_prioritizes_pinned(self, mem):
        # Save several memories
        mem.save("normal1", "Normal memory 1", memory_type=MEMORY_TYPE_LONG_TERM)
        mem.save("normal2", "Normal memory 2", memory_type=MEMORY_TYPE_LONG_TERM)
        mem.save("pinned1", "Pinned important memory", memory_type=MEMORY_TYPE_LONG_TERM,
                 tags=["pinned", "important"])

        result = mem.get_recent_for_prompt(count=10)
        assert len(result) == 3
        # Pinned should be first
        assert result[0]["key"] == "pinned1"


# ---------------------------------------------------------------------------
# Function Call Handler Tests
# ---------------------------------------------------------------------------

class TestFunctionCallHandler:
    """Test the handle_memory_function_call dispatcher."""

    @staticmethod
    def _make_fc(action, **kwargs):
        """Create a mock function call object."""
        from google.genai import types
        args = {"action": action}
        args.update(kwargs)
        return types.FunctionCall(id="test_id", name="memory", args=args)

    @pytest.fixture(autouse=True)
    def _patch_memory_system(self, mem, monkeypatch):
        """Replace the global memory_system with our test instance."""
        import src.memory as mem_mod
        monkeypatch.setattr(mem_mod, "memory_system", mem)

    def test_save_action(self):
        fc = self._make_fc("save", key="fc_key", content="fc content", category="test")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "ok"

    def test_read_action(self, mem):
        mem.save("read_key", "read content")
        fc = self._make_fc("read", key="read_key")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "ok"
        assert result.response["memory"]["content"] == "read content"

    def test_list_action(self, mem):
        mem.save("l1", "list item 1", memory_type=MEMORY_TYPE_LONG_TERM)
        mem.save("l2", "list item 2", memory_type=MEMORY_TYPE_SHORT_TERM)

        fc = self._make_fc("list")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "ok"
        assert result.response["count"] == 2

    def test_list_long_term_filter(self, mem):
        """Regression test: listing long_term should filter correctly."""
        mem.save("lt", "long term", memory_type=MEMORY_TYPE_LONG_TERM)
        mem.save("st", "short term", memory_type=MEMORY_TYPE_SHORT_TERM)

        fc = self._make_fc("list", memoryType="long_term")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "ok"
        assert result.response["count"] == 1

    def test_search_action(self, mem):
        mem.save("srch", "searchable content about dragons")
        fc = self._make_fc("search", searchTerm="dragon")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "ok"
        assert result.response["count"] >= 1

    def test_delete_action(self, mem):
        mem.save("del", "delete me")
        fc = self._make_fc("delete", key="del")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "ok"

    def test_stats_action(self, mem):
        mem.save("s1", "x")
        fc = self._make_fc("stats")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "ok"
        assert result.response["stats"]["total"] >= 1

    def test_pin_action(self, mem):
        mem.save("pin_me", "pin this")
        fc = self._make_fc("pin", key="pin_me", pin="true")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "ok"

        read = mem.read("pin_me")
        assert "pinned" in read["memory"]["tags"]

    def test_promote_action(self, mem):
        mem.save("promo", "promote me", memory_type=MEMORY_TYPE_SHORT_TERM)
        fc = self._make_fc("promote", key="promo", newType="long_term")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "ok"

        read = mem.read("promo")
        assert read["memory"]["memory_type"] == MEMORY_TYPE_LONG_TERM

    def test_unknown_action(self):
        fc = self._make_fc("explode")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "error"

    def test_save_missing_params(self):
        fc = self._make_fc("save")
        result = asyncio.get_event_loop().run_until_complete(handle_memory_function_call(fc))
        assert result.response["result"] == "error"


# ---------------------------------------------------------------------------
# Recall Sub-Agent Tests
# ---------------------------------------------------------------------------

class TestRecallAgent:
    """Test the recallMemories sub-agent (uses real API if key is available)."""

    @pytest.fixture(autouse=True)
    def _patch_memory_system(self, mem, monkeypatch):
        """Replace the global memory_system with our test instance."""
        import src.memory as mem_mod
        monkeypatch.setattr(mem_mod, "memory_system", mem)

    def test_recall_no_memories(self, mem):
        """Empty memory system should return graceful message."""
        result = asyncio.get_event_loop().run_until_complete(
            recall_memories("who is Alice?", api_key="fake_key")
        )
        assert result["result"] == "ok"
        assert result["count"] == 0

    def test_recall_no_api_key(self, mem):
        """Should fail gracefully without API key."""
        mem.save("x", "test")
        result = asyncio.get_event_loop().run_until_complete(
            recall_memories("test", api_key="")
        )
        assert result["result"] == "error"
        assert "API key" in result["message"]

    def test_recall_with_memories(self, mem):
        """Test recall with actual API call (skip if no key)."""
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            api_key = cfg.get("gemini", {}).get("api_key", "")
        except Exception:
            api_key = ""

        if not api_key or api_key.startswith("YOUR_"):
            pytest.skip("No API key configured")

        mem.save("alice_info", "Alice is 25 years old and loves gaming", category="personal")
        mem.save("alice_vrc", "Alice plays VRChat every weekend", category="personal")
        mem.save("bob_info", "Bob is a cat enthusiast", category="personal")

        result = asyncio.get_event_loop().run_until_complete(
            recall_memories("Tell me about Alice", api_key=api_key)
        )
        assert result["result"] == "ok"
        assert result["count"] == 3
        # The summary should mention Alice
        assert "alice" in result["summary"].lower() or "25" in result["summary"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
