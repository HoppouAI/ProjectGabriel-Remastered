# Avatar Setup Guide

End to end setup for getting ProjectGabriel's sensing layer onto your
avatar. Two pieces: the **Pose HUD** (screen-space coord strip) and the
**Sensor Rig** (VRCRaycast nav rays).

You'll do both through the **Tools > ProjectGabriel** menu in Unity. No
manual prefab building required.

> Unity version tested: **2022.3.22f1** (desktop / flatscreen VRChat).
> VR works too but the strip placement was tuned on flatscreen.

---

## Step 0 -- one-time install

1. Copy `unity_assets/` from this repo into your VRChat avatar project
   as `Assets/ProjectGabriel/`. Any folder path works, the editor scripts
   write their output under `Assets/ProjectGabriel/Generated/`.
2. Confirm you have the **VRChat Avatar SDK3** in the project (the rig
   builder needs `VRCRaycast` and `VRCExpressionParameters`).
3. **Optional but recommended:** install **VRCFury**. It's used to merge
   the sensor rig anchors and params into your avatar without touching
   the avatar hierarchy manually.
4. Wait for Unity to finish compiling. You should see two new menu items
   appear under **Tools > ProjectGabriel**:
   - Build Pose HUD
   - Build Sensor Rig

---

## Step 1 -- Pose HUD

This is the little color strip that encodes your world position. The
python side screen-captures it to know where you are.

### Build the prefab

1. Menu: **Tools > ProjectGabriel > Build Pose HUD**
2. The console prints `[GabrielPoseHudBuilder] built ...`.
3. Unity pings the new prefab in the Project window:
   `Assets/ProjectGabriel/Generated/GabrielPoseHUD.prefab`

### Drop it on your avatar

1. Drag `GabrielPoseHUD.prefab` onto your **avatar root** in the scene
   hierarchy. **Not** under the `Head` bone, **not** under `Hips`,
   **not** under `Armature`. Just the root GameObject that has your
   `VRCAvatarDescriptor` on it.

   > Why: VRChat scales the head bone to zero in first person to hide
   > head-mounted props. Anything parented under Head becomes invisible
   > to yourself. The avatar root stays visible. The vertex shader snaps
   > to NDC corners so the actual prefab transform doesn't matter, only
   > visibility does.

2. Leave its transform at `0, 0, 0` with identity rotation and scale
   `1, 1, 1`.

### Upload

That's it. Upload the avatar normally. When you wear it the strip will
appear in the **bottom-left corner** of your screen, ~272x16 px, using
pure red/green/blue/black pixels.

### Customizing position

If something in your HUD covers the bottom-left, you can shift the strip:

1. Select the `GabrielPoseHUD.prefab` asset
2. Find the `MeshRenderer` -> material slot, click it
3. The material exposes:
   - **Cell Size (px per logical pixel)** -- default 8
   - **Offset X (px from left)** -- default 0
   - **Offset Y (px from bottom)** -- default 0
4. Bump `Offset Y` up to ~50 to lift it above the VRChat HUD bar, etc.
5. **Important:** mirror your change in `src/pose_decoder.py`
   (the python side needs to know where to capture from).

### Privacy

Anyone whose camera renders your avatar also sees the strip. A
determined attacker who screen-captures their own view could decode
your world coords. Keep the cell size at the default 8 and turn off the
mesh renderer when you don't need the nav layer (Animator parameter
toggle works fine).

---

## Step 2 -- Sensor Rig (raycasts)

The python side uses VRCRaycast components for forward collision checks,
ledge detection, ceiling height, "what am I looking at", etc.

VRChat caps avatars at 80 raycasts total (shared with FinalIK). The
builder adds 11. Plenty of headroom.

### Build the prefab

1. Menu: **Tools > ProjectGabriel > Build Sensor Rig**
2. Console prints `Built sensor rig: ...`.
3. Two assets get created:
   - `Assets/ProjectGabriel/Generated/GabrielSensorRig.prefab`
   - `Assets/ProjectGabriel/Generated/GabrielSensorParameters.asset`

### Wire it onto your avatar

1. Drag `GabrielSensorRig.prefab` onto your **avatar root**. You'll see
   two children: `HeadAnchor` and `HipsAnchor`, each containing several
   `Ray_*` empties.

2. **Anchor the bones** (two ways, pick one):

   **A) With VRCFury (recommended)**
   - Select the `HeadAnchor` GameObject
   - Add Component -> **VRC Fury** -> Armature Link
   - Set **Link Mode** = Reparent
   - Drag your avatar's **Head** bone into the link target
   - Repeat on `HipsAnchor` with **Hips** as the target

   **B) Manual parenting**
   - Drag the `Ray_*` empties out of `HeadAnchor` and into your Head
     bone directly. Same for `Hips`. Then delete the now-empty anchor
     GameObjects.

3. **Merge the expression parameters** (this is what makes OSC actually
   publish the ray values):

   **A) With VRCFury (recommended)**
   - Select the `GabrielSensorRig` root
   - Add Component -> **VRC Fury** -> Full Controller
   - Leave Controller and Menu empty
   - Drag `GabrielSensorParameters.asset` into the **Parameters** list

   **B) Manual**
   - Open your avatar's Expression Parameters asset
   - Paste in each `<Ray>_Hit` (Bool), `<Ray>_Distance` (Float),
     `<Ray>_Ratio` (Float). All with **Synced = false**.
   - Avoids burning network bandwidth, the rays are local-only data.

4. **Animator params:** you do **not** need to add anything to the
   animator unless you want to drive animations from raycast hits.
   VRCRaycast publishes to OSC directly without animator wiring.

### Ray names + purposes

| Name        | Bone | Direction               | Length | Purpose                       |
|-------------|------|-------------------------|--------|-------------------------------|
| `Fwd`       | Head | forward                 | 5m     | front collision check         |
| `FwdNear`   | Hips | forward                 | 1.5m   | tight nav obstacle check      |
| `Left`      | Head | -90 yaw                 | 4m     | side clearance                |
| `Right`     | Head | +90 yaw                 | 4m     | side clearance                |
| `LeftFwd`   | Head | -45 yaw                 | 5m     | diagonal samples              |
| `RightFwd`  | Head | +45 yaw                 | 5m     | diagonal samples              |
| `Back`      | Head | 180 yaw                 | 3m     | rear awareness                |
| `DropFwd`   | Hips | forward + down 45 deg   | 3m     | ledge / stair detector        |
| `Down`      | Hips | straight down           | 2m     | floor distance                |
| `Up`        | Head | straight up             | 3m     | ceiling height                |
| `Gaze`      | Head | camera forward          | 30m    | "what is the AI looking at"   |

`Gaze` is configured with `Collision Mode = Hit Both` (worlds + players),
the rest are `Hit Worlds`. The builder sets this for you.

### Upload

Upload the avatar. Make sure **OSC** is enabled on the radial menu when
you wear it (Options -> OSC -> Enabled). Default ports 9000/9001 match
the python config.

---

## Step 3 -- verify it works

1. Run `python main.py` with your AI config pointed at VRChat
2. Wear the avatar in any VRChat world
3. Check the console:
   - `RaycastState` should log auto-discovered ray names like
     `Fwd`, `Left`, etc.
   - `PoseExfilReader` should log a non-zero pose `(x, y, z, yaw)`
4. Open the WebUI Mapping tab -- you should see the occupancy grid fill
   in around your avatar as you walk

If `PoseExfilReader` returns zero pose:
- The strip isn't on screen, or it's covered. Check the bottom-left
  corner. Heavy HUDs or world-side UI can cover it.
- Re-upload after confirming the prefab is on the avatar root, not the
  Head bone.

If the rays are publishing but values look wrong:
- Confirm each ray's empty GameObject has its **forward** axis pointing
  the right way (Unity gizmo, blue arrow). The builder pre-rotates them
  but custom edits can break this.
- Confirm the `Result` child transform exists under each ray. VRChat
  raycasts silently no-op without one. The builder adds them.

---

## What the builders do NOT do

- **VRCFury wiring** -- step 2 / step 3 of the sensor rig setup are
  manual because VRCFury's internal types shift across releases and
  reflection-based wiring was flaky. Two clicks each, not a big deal.
- **Avatar uploads** -- you still need to use the VRChat SDK Control
  Panel to build and upload.
- **Animator setup** -- not needed for OSC nav, only needed if you want
  to drive animations from raycast hits.
- **PoseHUD layer masks** -- the strip renders to every camera that sees
  your avatar (other players, mirrors). If you stream and want to hide
  it from your own OBS capture, crop it out in OBS rather than at the
  shader level. Coords are nav data, not secrets, but treat them like
  location data.

---

## Troubleshooting

**"Could not find shader 'ProjectGabriel/PoseExfilScreen'"**
The shader file didn't get imported. Confirm
`Assets/ProjectGabriel/shaders/PoseExfilScreen.shader` exists in the
Project window, then re-run the menu command.

**"VRCRaycast type not found"**
Avatar SDK3 isn't installed or is out of date. Install / update the
VRChat Creator Companion package "VRChat Avatars".

**Strip is at the top-left, not bottom-left**
You're on an older build. Re-import the shader. The fixed version checks
`_ProjectionParams.x` and uses `_ScreenParams.y - vertex.y` on the
desktop forward path.

**Strip flickers / changes color when other players walk by**
Confirm queue is `Overlay+5000` and `Blend Off` is on the SubShader.
Re-import the shader if not.

**OSC isn't publishing the ray params**
Skipped step 3 of the sensor rig setup. The params must be merged into
the avatar's Expression Parameters with `Synced = false`. Use VRCFury
Full Controller -> Parameters list, or paste them manually into the
avatar's `VRCExpressionParameters` asset.
