# Unity Assets for ProjectGabriel

Avatar side bits that let the python AI sense the VRChat world and know
where it is.

## What's in here

- `shaders/PoseExfilScreen.shader` -- screen-space pose strip. Renders a
  34x2 cell grid in the **bottom-left corner of the screen** encoding the
  avatar's world XYZ + forward vector. Python screen-captures it and
  decodes back to a `WorldPose`. See `src/pose_decoder.py` for the
  bit-packing format.
- `Editor/GabrielAvatarSetupWindow.cs` -- one-click installer window.
  Menu: **Tools > ProjectGabriel > Avatar Setup (Installer)**. Drag
  your avatar in, tick what to install, click the button.
- `Editor/GabrielPoseHudBuilder.cs` -- builds just the pose strip
  prefab. Menu: **Tools > ProjectGabriel > Build Pose HUD**. Used by
  the installer window, also usable on its own.
- `Editor/GabrielSensorRigBuilder.cs` -- builds just the VRCRaycast
  sensor rig prefab. Menu: **Tools > ProjectGabriel > Build Sensor
  Rig**.
- `AVATAR_SETUP.md` -- end-to-end avatar setup guide using the editor
  tools above.

## Quick start

1. Copy `unity_assets/` into your VRChat avatar project as
   `Assets/ProjectGabriel/`.
2. Make sure VRChat Avatar SDK3 is in the project (VRCFury optional,
   used for the sensor rig armature link + optional pose HUD toggle).
3. Wait for Unity to compile.
4. Open **Tools > ProjectGabriel > Avatar Setup (Installer)**, drag
   your avatar in, click install. Or follow `AVATAR_SETUP.md` for the
   full manual walkthrough including the VRCFury wiring.

## Python side that reads this stuff

- `src/raycast.py` -- VRCRaycast OSC state, auto-discovers ray names
- `src/pose_decoder.py` -- pose strip screen capture + decoder
- `src/spatial_map.py` -- occupancy grid, ray-to-world projection
- `src/pathfinder.py` -- A* on the occupancy grid
- `src/wanderer.py` -- autonomous map exploration using the above
