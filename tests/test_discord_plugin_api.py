"""Smoke tests for the discord-side plugin api surface.

Verifies that:
  - ctx.discord.register_tool stashes classes into the module registry
  - tools registered before the bot binds get attached when the handler
    binds late
  - tools registered after the bot binds get hot-attached
  - prompt contributors get collected
  - discord events get dispatched
  - send_system_instruction / send_user_text return False before the bot
    is up and route through inject_context after

Run directly:
    .venv\\Scripts\\python.exe tests/test_discord_plugin_api.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.plugins.api as plugin_api  # noqa: E402
from src.plugins.api import (  # noqa: E402
    Plugin,
    PluginContext,
    _bind_discord_session,
    collect_discord_prompt_contributions,
    emit_discord_event,
    iter_discord_tool_classes,
)


def reset():
    plugin_api._discord_tool_classes.clear()
    plugin_api._discord_prompt_contributors.clear()
    plugin_api._discord_event_subscribers.clear()
    plugin_api._discord_session = None
    plugin_api._discord_tool_handler = None


class FakeTool:
    def __init__(self, handler):
        self.handler = handler

    def declarations(self):
        return []

    async def handle(self, name, args):
        return None


class FakeHandler:
    def __init__(self):
        self.attached = []

    def register_plugin_tool(self, cls):
        self.attached.append(cls)


class FakeSession:
    def __init__(self):
        self._session = object()  # truthy
        self.injected = []

    async def inject_context(self, text):
        self.injected.append(text)


def make_ctx(name="t"):
    return PluginContext(name, config=None, plugin_dir=Path("."))


def test_register_tool_stashes_class():
    reset()
    ctx = make_ctx()
    ctx.discord.register_tool(FakeTool)
    assert FakeTool in iter_discord_tool_classes()
    print("ok register_tool_stashes_class")


def test_register_tool_dedupes():
    reset()
    ctx = make_ctx()
    ctx.discord.register_tool(FakeTool)
    ctx.discord.register_tool(FakeTool)
    assert iter_discord_tool_classes() == [FakeTool]
    print("ok register_tool_dedupes")


def test_late_bind_attaches_existing():
    reset()
    ctx = make_ctx()
    ctx.discord.register_tool(FakeTool)
    handler = FakeHandler()
    session = FakeSession()
    _bind_discord_session(session, handler)
    assert handler.attached == [FakeTool], f"expected late attach, got {handler.attached}"
    print("ok late_bind_attaches_existing")


def test_register_after_bind_hot_attaches():
    reset()
    handler = FakeHandler()
    session = FakeSession()
    _bind_discord_session(session, handler)
    ctx = make_ctx()
    ctx.discord.register_tool(FakeTool)
    assert handler.attached == [FakeTool], f"expected hot attach, got {handler.attached}"
    print("ok register_after_bind_hot_attaches")


def test_prompt_contributor_scoped_by_plugin():
    reset()
    a = make_ctx("plugin_a")
    b = make_ctx("plugin_b")
    a.discord.register_prompt_contributor("note", lambda: "from A")
    b.discord.register_prompt_contributor("note", lambda: "from B")
    out = collect_discord_prompt_contributions()
    assert "from A" in out and "from B" in out, f"got {out}"
    print("ok prompt_contributor_scoped_by_plugin")


def test_prompt_contributor_swallows_errors():
    reset()
    ctx = make_ctx()
    def boom():
        raise RuntimeError("broken")
    ctx.discord.register_prompt_contributor("ok", lambda: "fine")
    ctx.discord.register_prompt_contributor("bad", boom)
    out = collect_discord_prompt_contributions()
    assert out == ["fine"], f"errors should be skipped, got {out}"
    print("ok prompt_contributor_swallows_errors")


def test_event_subscribe_and_emit():
    reset()
    ctx = make_ctx()
    seen = []
    ctx.discord.subscribe("dm_received", lambda msg: seen.append(("sync", msg)))

    async def handler(msg):
        seen.append(("async", msg))

    ctx.discord.subscribe("dm_received", handler)

    async def driver():
        emit_discord_event("dm_received", "hello")
        # let async handler complete
        await asyncio.sleep(0.05)

    asyncio.run(driver())
    assert ("sync", "hello") in seen
    assert ("async", "hello") in seen, f"expected async to fire, got {seen}"
    print("ok event_subscribe_and_emit")


def test_send_returns_false_when_bot_offline():
    reset()
    ctx = make_ctx()

    async def driver():
        a = await ctx.discord.send_system_instruction("hi")
        b = await ctx.discord.send_user_text("hi")
        return a, b

    a, b = asyncio.run(driver())
    assert a is False and b is False, f"expected (False, False), got ({a}, {b})"
    print("ok send_returns_false_when_bot_offline")


def test_send_routes_to_inject_when_bound():
    reset()
    handler = FakeHandler()
    session = FakeSession()
    _bind_discord_session(session, handler)
    ctx = make_ctx()

    async def driver():
        a = await ctx.discord.send_system_instruction("be cool")
        b = await ctx.discord.send_user_text("hello there")
        return a, b

    a, b = asyncio.run(driver())
    assert a is True and b is True
    assert session.injected == ["SYSTEM INSTRUCTION: be cool", "hello there"], (
        f"injected wrong, got {session.injected}"
    )
    print("ok send_routes_to_inject_when_bound")


def test_main_session_unaffected_by_discord_calls():
    """Sanity: existing main-session registries are NOT touched by
    anything ctx.discord does. Backwards compat guarantee."""
    reset()
    plugin_api._tts_providers.clear()
    plugin_api._prompt_contributors.clear()
    plugin_api._event_subscribers.clear()
    plugin_api._chatbox_sources.clear()

    ctx = make_ctx()
    ctx.discord.register_tool(FakeTool)
    ctx.discord.register_prompt_contributor("x", lambda: "y")
    ctx.discord.subscribe("dm_received", lambda m: None)

    assert plugin_api._prompt_contributors == {}
    assert plugin_api._event_subscribers == {}
    assert plugin_api._chatbox_sources == {}
    print("ok main_session_unaffected_by_discord_calls")


def main():
    tests = [
        test_register_tool_stashes_class,
        test_register_tool_dedupes,
        test_late_bind_attaches_existing,
        test_register_after_bind_hot_attaches,
        test_prompt_contributor_scoped_by_plugin,
        test_prompt_contributor_swallows_errors,
        test_event_subscribe_and_emit,
        test_send_returns_false_when_bot_offline,
        test_send_routes_to_inject_when_bound,
        test_main_session_unaffected_by_discord_calls,
    ]
    for t in tests:
        t()
    print(f"\nall {len(tests)} discord plugin api tests passed")


if __name__ == "__main__":
    main()
