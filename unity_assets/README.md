# Unity Assets for ProjectGabriel

Avatar-side bits that let the python side sense the VRChat world.

## What's in here

- `shaders/PoseExfil.shader` -- renders an 8x1 pixel strip encoding the
  avatar's world XYZ + yaw. Python screen-captures it and decodes back to
  a `WorldPose`. See `src/pose_decoder.py` for the format.
- `AVATAR_SETUP.md` -- which VRCRaycast components to add and how to wire
  the pose shader onto a canvas.

## Status

Experimental. Branch `raycast-experiment`. Not on main, not pushed.

Python side already in tree and tested:
- `src/raycast.py` -- OSC listener state for VRCRaycast params
- `src/pose_decoder.py` -- pixel strip decoder + screen capture loop
- `src/spatial_map.py` -- occupancy grid + ray->world projection
- `tests/test_raycast_and_mapping.py` -- 33 tests, all green

## Order of operations to actually use this

1. Wire `RaycastState` into `VRChatClient`'s OSC dispatcher (TODO)
2. Add a few raycasts to a test avatar per `AVATAR_SETUP.md`
3. Add the pose shader quad to that avatar
4. Run, watch `RaycastState.get_all()` and `PoseExfilReader.get()` populate
5. Start `SpatialMapper` and confirm the grid fills out as you walk around
6. Build wanderer integration + AI tools

If any of the above flops, `git checkout main` and the whole experiment is
gone.
