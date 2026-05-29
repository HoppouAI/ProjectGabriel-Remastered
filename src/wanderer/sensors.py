"""Raycast sensor readers fed by VRChatOSC.raycast_state."""

from __future__ import annotations


class SensorsMixin:
    def _ray(self, name):
        state = getattr(self.osc, "raycast_state", None) if self.osc else None
        if state is None:
            return None
        r = state.get(name)
        if r is None or not r.is_fresh():
            return None
        return r

    def _forward_clearance(self):
        """Effective forward clearance in meters. Prefers FwdNear (1.5m hip
        ray) for stopping, falls back to Fwd (5m head ray). Returns None if
        neither ray is reporting yet."""
        near = self._ray("FwdNear")
        head = self._ray("Fwd")
        if near is not None and near.hit:
            return near.distance
        if head is not None and head.hit:
            return head.distance
        if near is not None or head is not None:
            # no hit on either = wide open, return a sensible large value
            if head is not None:
                return max(head.distance, 5.0)
            return max(near.distance, 1.5)
        return None

    def _side_blocked(self, name, threshold):
        r = self._ray(name)
        if r is None:
            return False
        return r.hit and r.distance < threshold

    def _side_distance(self, name, default=None):
        r = self._ray(name)
        if r is None:
            return default
        # for steering we treat "no hit" as max reference distance
        if not r.hit:
            return max(r.distance, self._cfg["side_reference"])
        return r.distance

    def _drop_ahead(self):
        r = self._ray("DropFwd")
        if r is None:
            return False
        if r.hit:
            self._dropfwd_ever_hit = True
            return False
        # only trust "miss = ledge" once we have proof the ray works,
        # otherwise a bad ray config would lock us in reverse forever
        return self._dropfwd_ever_hit
