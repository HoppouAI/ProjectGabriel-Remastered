# Unity Editor Scripts

Three editor-only entry points under **Tools > ProjectGabriel** in the
menu bar:

| Menu item                       | Script                              | What it does                                                          |
|---------------------------------|-------------------------------------|-----------------------------------------------------------------------|
| Avatar Setup (Installer)        | `GabrielAvatarSetupWindow.cs`       | EditorWindow. Drag avatar, tick what to install, click button.        |
| Build Pose HUD                  | `GabrielPoseHudBuilder.cs`          | Builds just the pose strip prefab (manual workflow).                  |
| Build Sensor Rig                | `GabrielSensorRigBuilder.cs`        | Builds just the VRCRaycast sensor rig prefab (manual workflow).       |

All prefabs land in `Assets/ProjectGabriel/Generated/`.

For the full step-by-step avatar setup (which is what you actually want
to read), see `unity_assets/AVATAR_SETUP.md`. The notes below are just
for understanding what each script is doing under the hood.

---

## Avatar Setup (Installer)

EditorWindow that ties the two builders together. Workflow:

1. Drag your avatar's GameObject into the window's avatar slot. The
   window walks up parents to find the first ancestor with a
   `VRCAvatarDescriptor`, so dragging any child of the avatar works.
2. Tick **Pose HUD** and/or **Sensor Rig**.
3. (Optional) tick **Replace existing instances** so old copies get
   nuked first. Leave off if you've hand-edited the existing prefab
   instances on the avatar.
4. Click **Install on '<avatar>'**.

The window calls `GabrielPoseHudBuilder.BuildPrefab()` and
`GabrielSensorRigBuilder.BuildPrefab()` to refresh the prefabs on disk,
then `PrefabUtility.InstantiatePrefab` to parent fresh instances onto
the avatar at identity transforms. Everything is wrapped in a single
Undo group so Ctrl+Z reverts the whole install.

---

## Build Pose HUD

Creates `GabrielPoseHUD.prefab` containing a single quad with the
`ProjectGabriel/PoseExfilScreen` material.

Key details:
- Quad mesh is `+-0.5` in object space, but its `Bounds` are forced to
  cover roughly the entire avatar (feet to above head, ~3m wide) so the
  camera is always inside the renderer AABB and Unity never frustum-culls
  the strip.
- The shader's vert pass snaps every vertex to NDC corners via
  `sign(vertex.xy)`, so the actual quad transform doesn't affect what
  gets drawn. You can parent / move / rotate the prefab freely.
- Material exposes three floats: `_CellSize`, `_OffsetX`, `_OffsetY`.

---

## Build Sensor Rig

Creates two assets:
- `GabrielSensorRig.prefab` -- root with `HeadAnchor` and `HipsAnchor`
  children, each containing pre-rotated `Ray_*` empties. Each ray empty
  has a `VRCRaycast` component plus a `Result` child transform (VRChat
  silently refuses to publish raycast data without a `Result`).
- `GabrielSensorParameters.asset` -- a `VRCExpressionParameters` with
  every `<Ray>_Hit` / `_Distance` / `_Ratio` declared as unsynced.

The script uses **reflection** to talk to `VRCRaycast` and
`VRCExpressionParameters` so it compiles even when the SDK isn't
present. You just get a warning + no output in that case.

### What it deliberately skips

- VRCFury setup -- the Armature Link components on the anchors and the
  optional local-only Toggle on the pose HUD are still manual. VRCFury's
  internal types change between releases, so reflection-based wiring
  proved too brittle. Two clicks per component, see `AVATAR_SETUP.md`.
- Animator wiring -- VRCRaycast publishes to OSC directly, no animator
  parameters needed for the python side to read the rays.

---

## Troubleshooting compile errors

Both scripts wrap themselves in `#if UNITY_EDITOR ... #endif` so they
will not ship in player builds. If you see compile errors:

- **"VRCRaycast type not found"** -- Avatar SDK3 missing or outdated.
- **"VRCExpressionParameters type not found"** -- same fix.
- Any other error usually means a stale .meta file. Close Unity, delete
  `Library/`, reopen.
