"""Mapping + waypoint orchestration for the main webUI.

Wraps PoseExfilReader, VoxelNavManager, VoxelExplorer and WaypointStore
into one easy-to-poke service. The webUI hits this; nothing autostarts.

Designed to be created once in main.py and shoved into control_server's
shared_state under "mapping_service".

Implementation is split into:
    _base.py     -- MappingServiceBase + _RegionGuess
    lifecycle.py -- start/stop/explore toggle + the tick thread
    state.py     -- get_state/get_world_cells/follow_status
    waypoints.py -- list/add/remove waypoints
    editing.py   -- WebUI cell edits + stray/jump-artifact cleanup
    worlds.py    -- update_settings/list_worlds/delete_world
    navigation.py-- pathfind preview + goto + alignment
    manual.py    -- manual mapping (raycast walls + grid lock)
    service.py   -- composes the mixins
"""

from .service import MappingService

__all__ = ["MappingService"]
