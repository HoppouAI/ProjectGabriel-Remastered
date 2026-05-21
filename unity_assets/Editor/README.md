# Unity Editor Scripts

Two editor-only one-click builders under **Tools > ProjectGabriel** in
the menu bar:

| Menu item                  | Script                          | What it builds                                    |
|----------------------------|---------------------------------|---------------------------------------------------|
| Build Pose HUD             | `GabrielPoseHudBuilder.cs`      | Screen-space pose strip prefab (bottom-left)      |
| Build Sensor Rig           | `GabrielSensorRigBuilder.cs`    | VRCRaycast sensor rig prefab + params asset       |

Both scripts write their output to
`Assets/ProjectGabriel/Generated/`.

For the full step-by-step avatar setup (which is what you actually want
to read), see `unity_assets/AVATAR_SETUP.md`. The notes below are just
for understanding what each script is doing under the hood.

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

- VRCFury setup -- VRCFury's internal types change between releases, so
  the script leaves the Armature Link and Full Controller wiring as
  manual steps (two clicks each, see `AVATAR_SETUP.md`).
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
