# Avatar Raycast Layout

This is the suggested set of `VRCRaycast` components to add to the avatar
so the python side can do navigation, pathfinding, and spatial awareness.

VRChat caps raycasts at 80 per avatar (shared with FinalIK), so we use
about a dozen. Plenty of headroom.

## Naming convention

Parameter prefix for each raycast goes into Animator + Expression
Parameters. VRChat appends `_Hit`, `_Distance`, `_Ratio` automatically.

In the Expression Parameters list set them all to **synced = false** to
save bandwidth (we only need them locally for OSC).

## Rays

| Name        | Origin           | Direction              | Length | Purpose                                  |
|-------------|------------------|------------------------|--------|------------------------------------------|
| `Fwd`       | head             | forward                | 5m     | front collision check, follow safety     |
| `FwdNear`   | hip              | forward                | 1.5m   | tight obstacle avoidance while walking   |
| `Left`      | head             | left (-90 yaw)         | 4m     | side clearance for wanderer              |
| `Right`     | head             | right (+90 yaw)        | 4m     | side clearance                            |
| `LeftFwd`   | head             | forward-left (-45 yaw) | 5m     | diagonal pathfinding samples              |
| `RightFwd`  | head             | forward-right (+45 yaw)| 5m     | diagonal pathfinding samples              |
| `Back`      | head             | back (180 yaw)         | 3m     | know when something's behind you          |
| `DropFwd`   | hip              | forward + down 45 deg  | 3m     | ledge/stair detector ahead                |
| `Down`      | hip              | straight down          | 2m     | floor distance, prone detection           |
| `Up`        | head             | straight up            | 3m     | ceiling height, awareness of overhead     |
| `Gaze`      | head             | camera forward         | 30m    | "what is Gabriel looking at"              |

The Gaze ray is the long one. The rest are short for navigation.

Add a "ReachRight" ray if you want grab/use targeting feedback later.

## Setup steps in Unity

1. On each VRCRaycast component:
   - **Collision Mode**: `Hit Worlds` for nav rays, `Hit Both` for `Gaze`
   - **Origin**: a small empty transform parented under the named body part
   - **Direction**: relative to its parent (use the transform's forward)
   - **Parameter**: matches the Name column above
   - **Behavior on Miss**: `Snap To End`
   - **Result Transform**: a child empty named `Result` (REQUIRED, the
     component throws a red warning and does nothing without one)
2. Add matching params to the Animator (`<Name>_Hit` Bool,
   `<Name>_Distance` Float, `<Name>_Ratio` Float)
3. Add matching params to Expression Parameters with `synced = false`.
   If you used the editor script you get a `GabrielSensorParameters`
   asset, the easiest way to merge it into the avatar is a **VRCFury >
   Full Controller** component on the rig root with that asset dropped
   into the **Parameters** list. Without this step VRChat never publishes
   the params over OSC even though they exist on the raycast components.
4. Enable OSC in VRChat, default port 9000 in / 9001 out, listener will pick
   them up.

## How the python side reads them

`src/raycast.py :: RaycastState` is wired into the existing OSC dispatcher
in `src/vrchat.py`. It auto-discovers ray names from the incoming OSC
addresses, no per-ray code needed.

For mapping, `src/spatial_map.py :: RayConfig` per ray tells the mapper the
heading + max range so it can project hits into world coordinates using
the avatar pose decoded from the `PoseExfil` shader strip.

## Pose strip placement

The `unity_assets/shaders/PoseExfil.shader` quad needs to be on screen at
a known location. The python decoder defaults to top-left, 8x1 pixels.

Cleanest setup:
- Add a Canvas in `Screen Space - Camera` mode under the main camera
- Add a RawImage child, size = 8x1 px, anchor top-left, position (0, 0)
- Assign a material using `ProjectGabriel/PoseExfil`
- Set the canvas layer to one only the local camera sees (NOT the mirror
  / NOT other players) so you don't broadcast your coords to the lobby

For streaming, crop the 8x1 strip out of OBS or move it behind a UI panel
the cropping mask covers.
