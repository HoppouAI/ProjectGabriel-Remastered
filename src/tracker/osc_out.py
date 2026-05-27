"""OSC output, sends look/move/sprint axes to VRChat."""


class OscMixin:
    def _send_osc(self):
        """Send smoothed axis values to VRChat. All axes are float -1..+1."""
        if not self.osc:
            return

        client = self.osc.client
        cfg = self._cfg
        dz = cfg["deadzone"]

        look_h = max(-1.0, min(1.0, self._smoothed_look_h))
        look_v = max(-1.0, min(1.0, self._smoothed_look_v))
        forward = max(-1.0, min(1.0, self._smoothed_forward))

        if abs(look_h) < dz:
            look_h = 0.0
        client.send_message("/input/LookHorizontal", float(look_h))

        if abs(look_v) < dz:
            look_v = 0.0
        client.send_message("/input/LookVertical", float(look_v))

        client.send_message("/input/Vertical", float(forward))
        client.send_message("/input/Run", 1 if self._sprinting else 0)

        # strafe only when heavily off-centre
        if abs(look_h) > cfg["strafe_threshold"]:
            strafe = max(-1.0, min(1.0, look_h * cfg["strafe_scale"]))
            client.send_message("/input/Horizontal", float(strafe))
        else:
            client.send_message("/input/Horizontal", 0.0)

    def _zero_osc(self):
        """Reset all axes to zero, called on shutdown."""
        if not self.osc:
            return
        client = self.osc.client
        client.send_message("/input/LookHorizontal", 0.0)
        client.send_message("/input/LookVertical", 0.0)
        client.send_message("/input/Vertical", 0.0)
        client.send_message("/input/Horizontal", 0.0)
        client.send_message("/input/Run", 0)
