"""Tests for the navigation <-> motion puppet handoff.

The puppet used to write 0 to Vertical/LookHorizontal every frame, and the
explorer dedupes against its own last sent value, so the avatar silently
stopped mid path whenever motion was enabled.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import src.motion_client as mc


class FakeOsc:
    def __init__(self):
        self.sent = []

    def send_message(self, addr, value):
        self.sent.append((addr, value))

    def addrs(self):
        return [a for a, _ in self.sent]


@pytest.fixture(autouse=True)
def _clear_arbiter():
    mc._nav_until = 0.0
    mc._CLIENT = None
    yield
    mc._nav_until = 0.0
    mc._CLIENT = None


def _client(osc, nav_mode="pause"):
    c = mc.MotionClient(osc, "127.0.0.1", 8765, nav_mode=nav_mode)
    c._active = True
    c._got_frame = True
    c.backend = "ardy"
    return c


def test_arbiter_expires_so_a_missed_stop_cant_wedge_the_body():
    assert not mc.navigation_driving()
    mc.navigation_tick()
    assert mc.navigation_driving()
    mc._nav_until = time.monotonic() - 0.01
    assert not mc.navigation_driving()


def test_pause_mode_drops_the_puppet_without_zeroing_move_inputs():
    osc = FakeOsc()
    c = _client(osc)
    mc._CLIENT = c
    mc.navigation_tick()

    osc.sent.clear()
    asyncio.run(c._nav_update(time.monotonic()))
    assert c._navigating
    assert not c._active
    assert (mc.PREFIX + "Enable", False) in osc.sent
    # the whole point: never touch the channels the navigator is using
    assert "/input/Vertical" not in osc.addrs()
    assert "/input/LookHorizontal" not in osc.addrs()


def test_puppet_comes_back_once_navigation_lets_go():
    osc = FakeOsc()
    c = _client(osc)
    mc._CLIENT = c
    mc.navigation_tick()
    asyncio.run(c._nav_update(time.monotonic()))
    assert not c._active

    mc._nav_until = 0.0
    osc.sent.clear()
    asyncio.run(c._nav_update(time.monotonic()))
    assert not c._navigating
    assert c._active
    assert (mc.PREFIX + "Enable", True) in osc.sent


def test_model_mode_walks_in_place_without_driving_him():
    osc = FakeOsc()
    c = _client(osc, nav_mode="model")
    mc._CLIENT = c
    assert c.walks_for_navigation

    sent = []

    async def fake_send(obj):
        sent.append(obj)

    async def fake_connect(timeout=8.0):
        pass

    c._send = fake_send
    c.ensure_connected = fake_connect
    mc.navigation_tick()
    asyncio.run(c._nav_update(time.monotonic()))
    assert c._navigating
    assert c._active  # puppet stays up so the body strides
    assert any(m.get("type") == "prompt" for m in sent)
    # root motion must NOT reach the move inputs, navigation owns those
    assert not c._locomotion


def test_model_mode_needs_ardy():
    osc = FakeOsc()
    c = _client(osc, nav_mode="model")
    c.backend = "dart"
    mc._CLIENT = c
    assert not c.walks_for_navigation


def test_expression_layer_stands_down_while_navigating():
    from src.motion_expression import MotionExpression

    exp = MotionExpression.__new__(MotionExpression)
    exp._suppressed = False
    exp._speaking = True
    exp._talking = ["a person talks while gesturing with both hands"]
    exp._thinking = False
    exp._thinking_prompt = ""
    exp._last_activity = time.time()
    exp._release_after = 180.0
    exp._idle_after = 25.0
    exp._idle = []

    assert exp._want() == "talk"
    mc.navigation_tick()
    assert exp._want() is None
