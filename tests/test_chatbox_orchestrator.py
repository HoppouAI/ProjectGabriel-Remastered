"""Tests for src.gemini_live.chatbox_orchestrator.ChatboxOrchestrator.

Run directly with:
    .venv\\Scripts\\python.exe tests/test_chatbox_orchestrator.py
"""
import sys
import time
from pathlib import Path

# Make `src` importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gemini_live.chatbox_orchestrator import (  # noqa: E402
    ChatboxOrchestrator,
    _FORCE_REFRESH_SECONDS,
)
import src.plugins.api as plugin_api  # noqa: E402


class FakeSource:
    def __init__(self, active=False, text="hello", crash_active=False, crash_render=False):
        self.active = active
        self.text = text
        self.crash_active = crash_active
        self.crash_render = crash_render
        self.cleared = 0

    def is_active(self):
        if self.crash_active:
            raise RuntimeError("active boom")
        return self.active

    def render(self):
        if self.crash_render:
            raise RuntimeError("render boom")
        return self.text

    def on_clear(self):
        self.cleared += 1


def make_orchestrator():
    sent = []

    def send(text):
        sent.append(text)

    orch = ChatboxOrchestrator(send_chatbox=send)
    return orch, sent


def reset_plugin_registries():
    plugin_api._chatbox_sources.clear()


def test_dedupes_same_text():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    src = FakeSource(active=True, text="same")
    plugin_api._chatbox_sources["s1"] = (100, src)
    orch.tick()
    orch.tick()
    orch.tick()
    assert sent == ["same"], f"expected one send, got {sent}"
    print("ok dedupes_same_text")


def test_force_refresh_after_timeout():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    src = FakeSource(active=True, text="same")
    plugin_api._chatbox_sources["s1"] = (100, src)
    orch.tick()
    # Backdate so the next tick should force-refresh
    orch._last_send -= _FORCE_REFRESH_SECONDS + 1
    orch.tick()
    assert sent == ["same", "same"], f"expected force refresh, got {sent}"
    print("ok force_refresh_after_timeout")


def test_text_change_triggers_send():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    src = FakeSource(active=True, text="a")
    plugin_api._chatbox_sources["s1"] = (100, src)
    orch.tick()
    src.text = "b"
    orch.tick()
    assert sent == ["a", "b"], f"expected ab, got {sent}"
    print("ok text_change_triggers_send")


def test_priority_order():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    low = FakeSource(active=True, text="low_pri")
    high = FakeSource(active=True, text="high_pri")
    plugin_api._chatbox_sources["high"] = (10, high)
    plugin_api._chatbox_sources["low"] = (200, low)
    orch.tick()
    assert sent == ["high_pri"], f"expected lowest priority value to win, got {sent}"
    print("ok priority_order")


def test_winner_change_fires_on_clear():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    a = FakeSource(active=True, text="A")
    b = FakeSource(active=False, text="B")
    plugin_api._chatbox_sources["a"] = (50, a)
    plugin_api._chatbox_sources["b"] = (60, b)
    orch.tick()
    assert sent == ["A"]
    a.active = False
    b.active = True
    orch.tick()
    assert a.cleared == 1, f"a should have been cleared once, got {a.cleared}"
    assert sent == ["A", "B"], f"expected A then B, got {sent}"
    print("ok winner_change_fires_on_clear")


def test_no_winner_calls_on_clear_and_stops_writing():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    a = FakeSource(active=True, text="A")
    plugin_api._chatbox_sources["a"] = (50, a)
    orch.tick()
    assert sent == ["A"]
    a.active = False
    orch.tick()
    assert a.cleared == 1, f"a should be cleared once when going inactive, got {a.cleared}"
    # No further sends -- idle banner takes over from another loop, we
    # do NOT spam empty strings.
    assert sent == ["A"], f"expected no extra send when going idle, got {sent}"
    # Subsequent ticks remain quiet
    orch.tick()
    orch.tick()
    assert sent == ["A"]
    assert a.cleared == 1, "on_clear must only fire once per transition"
    print("ok no_winner_calls_on_clear_and_stops_writing")


def test_is_active_exception_does_not_crash():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    bad = FakeSource(crash_active=True)
    good = FakeSource(active=True, text="good")
    plugin_api._chatbox_sources["bad"] = (10, bad)
    plugin_api._chatbox_sources["good"] = (20, good)
    orch.tick()
    assert sent == ["good"], f"good source should still win, got {sent}"
    print("ok is_active_exception_does_not_crash")


def test_render_exception_falls_through():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    bad = FakeSource(active=True, crash_render=True)
    good = FakeSource(active=True, text="good")
    plugin_api._chatbox_sources["bad"] = (10, bad)
    plugin_api._chatbox_sources["good"] = (20, good)
    orch.tick()
    assert sent == ["good"], f"expected fallthrough to good, got {sent}"
    print("ok render_exception_falls_through")


def test_source_suspended_after_repeated_failures():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    bad = FakeSource(crash_active=True)
    plugin_api._chatbox_sources["bad"] = (10, bad)
    for _ in range(10):
        orch.tick()
    assert "bad" in orch._suspended, "expected bad source to be suspended"
    print("ok source_suspended_after_repeated_failures")


def test_builtin_beats_plugin():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    plugin = FakeSource(active=True, text="plugin")
    plugin_api._chatbox_sources["p"] = (1, plugin)

    def builtin():
        return ("music", "now playing")

    orch.register_builtin("music", builtin)
    orch.tick()
    assert sent == ["now playing"], f"builtin should always win, got {sent}"
    print("ok builtin_beats_plugin")


def test_has_active_source_reflects_state():
    reset_plugin_registries()
    orch, _sent = make_orchestrator()
    src = FakeSource(active=False)
    plugin_api._chatbox_sources["s"] = (100, src)
    assert orch.has_active_source() is False
    src.active = True
    assert orch.has_active_source() is True
    print("ok has_active_source_reflects_state")


def test_render_returning_none_falls_through():
    reset_plugin_registries()
    orch, sent = make_orchestrator()
    a = FakeSource(active=True, text=None)
    b = FakeSource(active=True, text="b")
    plugin_api._chatbox_sources["a"] = (10, a)
    plugin_api._chatbox_sources["b"] = (20, b)
    orch.tick()
    assert sent == ["b"], f"None render should fall through, got {sent}"
    print("ok render_returning_none_falls_through")


def main():
    tests = [
        test_dedupes_same_text,
        test_force_refresh_after_timeout,
        test_text_change_triggers_send,
        test_priority_order,
        test_winner_change_fires_on_clear,
        test_no_winner_calls_on_clear_and_stops_writing,
        test_is_active_exception_does_not_crash,
        test_render_exception_falls_through,
        test_source_suspended_after_repeated_failures,
        test_builtin_beats_plugin,
        test_has_active_source_reflects_state,
        test_render_returning_none_falls_through,
    ]
    for t in tests:
        t()
    print(f"\nall {len(tests)} chatbox orchestrator tests passed")


if __name__ == "__main__":
    main()
