# Sensor Rig Builder (Unity editor script)

`Editor/GabrielSensorRigBuilder.cs` is an editor-only script that builds a
drag-and-drop sensor rig prefab for the avatar in one click.

## Install

1. Copy `unity_assets/` into your VRChat avatar Unity project as
   `Assets/ProjectGabriel/` (any path is fine, the script puts its output
   under `Assets/ProjectGabriel/Generated/`).
2. Make sure your project has the **VRChat Avatar SDK3** (for `VRCRaycast`
   and `VRCExpressionParameters`). VRCFury is optional, it's only used
   when you add Armature Link components by hand to the anchors.
3. Wait for Unity to compile the script.

## Use

1. Menu bar: **Tools > ProjectGabriel > Build Sensor Rig**
2. Watch the Console for `Built sensor rig: ...`. The selected prefab
   appears in `Assets/ProjectGabriel/Generated/GabrielSensorRig.prefab`.
3. Drag that prefab into your avatar's root.
4. On the `HeadAnchor` child, add a VRCFury component, pick **Armature
   Link**, set the link target bone to **Head**. Repeat on `HipsAnchor`
   with **Hips**.
5. On the `GabrielSensorRig` root, add a **VRCFury > Full Controller**
   component. Leave Controller and Menu empty, then drag
   `GabrielSensorParameters.asset` into the **Parameters** list. Without
   this step VRChat never publishes the `<Ray>_*` params over OSC.
6. Upload. VRCFury reparents each anchor under its bone and merges the
   params into your avatar's expression parameters at build time.

## What it generates

- `GabrielSensorRig.prefab` -- root with `HeadAnchor` + `HipsAnchor`
  children, each containing the `Ray_*` empties pre-rotated and with
  `VRCRaycast` components set up per `AVATAR_SETUP.md`. Each ray also
  has a `Result` child transform that VRC writes the hit point into.
- `GabrielSensorParameters.asset` -- a `VRCExpressionParameters` asset
  with all `<Ray>_Hit`/`_Distance`/`_Ratio` parameters declared as
  unsynced so they only fire local OSC.

## What it does NOT do

- Wire up VRCFury for you. Steps 4 and 5 above are manual because the
  internal VRCFury types change shape between releases and reflection
  was unreliable.
- Add the `PoseExfil` shader quad to your avatar. That's a separate
  manual setup (see `AVATAR_SETUP.md` "Pose strip placement").

## If the script complains

- "VRCRaycast type not found" -- update the VRChat SDK, raycasts shipped
  in recent 3.x.
- "VRCExpressionParameters type not found" -- VRChat SDK missing. Install
  the Avatars SDK.
