"""OSC output: movement, stop, jump."""

from __future__ import annotations

import time


class OscMixin:
    def _send_osc(self, turn, forward):
        if not self.osc or self._paused:
            return
        from src.motion_client import navigation_tick
        navigation_tick()
        client = self.osc.client
        client.send_message("/input/LookHorizontal", float(max(-1, min(1, turn))))
        client.send_message("/input/Vertical", float(max(-1, min(1, forward))))
        client.send_message("/input/Horizontal", 0.0)
        client.send_message("/input/Run", 0)

    def _zero_osc(self):
        if not self.osc:
            return
        client = self.osc.client
        client.send_message("/input/LookHorizontal", 0.0)
        client.send_message("/input/LookVertical", 0.0)
        client.send_message("/input/Vertical", 0.0)
        client.send_message("/input/Horizontal", 0.0)
        client.send_message("/input/Run", 0)

    def _do_jump(self):
        if not self.osc:
            return
        client = self.osc.client
        client.send_message("/input/Jump", 1)
        time.sleep(0.05)
        client.send_message("/input/Jump", 0)
