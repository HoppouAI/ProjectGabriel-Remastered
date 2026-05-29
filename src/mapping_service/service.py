"""The actual MappingService class. Just glues the mixins together."""

from __future__ import annotations

from ._base import MappingServiceBase
from .editing import EditingMixin
from .lifecycle import LifecycleMixin
from .manual import ManualMappingMixin
from .navigation import NavigationMixin
from .state import StateMixin
from .waypoints import WaypointsMixin
from .worlds import WorldsMixin


class MappingService(
    LifecycleMixin,
    StateMixin,
    WaypointsMixin,
    EditingMixin,
    WorldsMixin,
    NavigationMixin,
    ManualMappingMixin,
    MappingServiceBase,
):
    """Owns the mapping subsystem. All public methods are thread safe.

    Lifecycle:
        ms = MappingService(osc, instance_monitor=im)
        ms.start(explore=True)      # webUI clicks Start
        ms.add_waypoint("couch")    # adds at current pose
        ms.stop()                   # webUI clicks Stop

    Each call to start() rescans for the pose strip on the screen, so the
    user can move the strip around between sessions.
    """
