# Motion Server

Real time full body motion generation for Gabriel. Type a text prompt, a
diffusion model ([DART](https://github.com/zkf1997/DART)) generates human
motion frame by frame, and the frames stream over a websocket into VRChat as
avatar puppet parameters. He dances, sits, bows, lies down, waves, and
walking motions physically move him through the world.

This README documents the ENTIRE system, including the parts that live
outside this repo (the Unity rig), so everything can be rebuilt from nothing.

## Architecture

```
prompt ("a person dances energetically")
   |
   v
DART rollout engine (server.py, GPU)          motion primitives, 8 future
   |   SMPL body frames at model fps          frames per denoise step
   v
retarget.py                                   SMPL world frames -> 29 FBT
   |   muscle params + locomotion velocities  params + body frame velocity
   v
websocket (port 8765, json frames)
   |
   v
src/motion_client.py (main app)               60hz OSC sender, exponential
   |   smoothing, locomotion mapping          smoothing tau 0.07s
   v
VRChat OSC (/avatar/parameters/FBT/*, /input/Vertical ...)
   |
   v
DesktopFBT puppet rig on the avatar           direct blend tree + VRC
                                              rotation constraint
```

Two model variants, selectable with `--model`:

| | `babel` (default) | `hml3d` |
|---|---|---|
| prompts | bare action verbs ("dance", "sit down") | full sentences ("a person dances energetically") |
| fps | 30 | 20 |
| bodies | SMPL-X | SMPL-H |
| sampling | ddim5 (full 10 step cant hold 30fps) | full 10 step (trained config) |
| quality | okay | noticeably better, use this one |

The server normalizes incoming prompts to whichever style its model was
trained on (wraps terse verbs in "a person ..." for hml3d, strips the
narration for babel), so the Gemini tool always writes sentences.

## Setup from nothing

### 1. venv

Python 3.10 venv in `motion_server/.venv`. From the repo root:

```
bin\uv.exe venv motion_server\.venv --python 3.10
bin\uv.exe pip install --python motion_server\.venv\Scripts\python.exe -r motion_server\requirements.txt
```

Gotchas learned the hard way:
- torch must be `2.5.1+cu121` from the cu121 index. Pin the `+cu121` local
  tag explicitly or a later `uv pip install` will clobber it with the cpu wheel.
- `numpy<2` (DART pickles and chumpy both break on 2.x).
- pytorch3d is NOT installed; `shims/pytorch3d/` is a pure torch
  reimplementation of the transform functions DART uses. server.py puts
  `shims/` on sys.path ahead of everything.

### 2. DART code

```
git clone https://github.com/zkf1997/DART motion_server/DART
```

`motion_server/DART/` is gitignored (vendor code + big weights). Never edit
DART files, all Windows compat is runtime patching inside server.py, in this
order (order matters):

1. sys.path: `shims/` first, then `DART/`
2. chdir into `DART/` (relative paths everywhere in their code)
3. numpy alias shims (`np.float`, `np.int`, `np.bool`, `np.object`, `np.str`)
4. `pathlib.PosixPath = WindowsPath` on nt (their checkpoints/yamls were
   pickled on linux and contain PosixPath objects)

### 3. Checkpoints and data (google drive)

Sources are linked from the DART repo README. gdown rate limits constantly;
this direct URL form works with plain curl:

```
curl.exe -L -o <file> "https://drive.usercontent.google.com/download?id=<FILE_ID>&export=download&confirm=t"
```

To list a drive FOLDER without gdown: curl the folder URL and regex
`data-id="..."` + `data-tooltip="..."` pairs out of the HTML.

Required layout under `motion_server/DART/` (sizes for sanity checking):

```
mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt   78.5 MB  babel denoiser
mld_denoiser/mld_fps_clip_repeat_euler/args.yaml
mld_denoiser/smplh_hml3d_2_8_4/checkpoint_300000.pt           78.5 MB  hml3d denoiser
mld_denoiser/smplh_hml3d_2_8_4/args.yaml
mvae/mvae_fps_clip/checkpoint_200000.pt                       47.3 MB  babel vae
mvae/mvae_fps_clip/args.yaml
mvae/mvae_smplh_hml3d_2_8_4/checkpoint_200000.pt              47.3 MB  hml3d vae
mvae/mvae_smplh_hml3d_2_8_4/args.yaml
data/seq_data_zero_male/mean_std_h2_f8.pkl                             babel stats
data/seq_data_zero_male/train_text_embedding_dict.pkl        10.8 MB
data/hml3d_smplh/seq_data_zero_male/mean_std_h2_f8.pkl                 hml3d stats
data/hml3d_smplh/seq_data_zero_male/train_text_embedding_dict.pkl  61.3 MB
data/stand.pkl                                                         babel seed pose
data/stand_20fps.pkl                                                   hml3d seed pose
config_files/config_hydra/motion_primitive/*.yaml                      from the repo itself
```

Two yaml file identification tricks when drive names get mangled: `args.yaml`
starts with `# tyro YAML.` and is the larger one; `args_read.yaml` is a plain
dump.

### 4. Body models (registration walls)

**SMPL-X** (babel): register at https://smpl-x.is.tue.mpg.de, download the
"SMPL-X with locked head" (smplx_lockedhead) archive, extract so this exists:

```
DART/data/smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_MALE.npz
```

**SMPL-H** (hml3d): needs a SEPARATE registration at
https://mano.is.tue.mpg.de. The "Extended SMPL+H" (`smplh.tar.xz`) archive
only has npz bodies WITHOUT the MANO hand PCA keys that the smplx python
package requires unconditionally, so you need `mano_v1_2.zip` too:

1. download `smplh.tar.xz`, extract to `DART/smplh_extract/` (male/female/neutral/model.npz)
2. download `mano_v1_2.zip`, extract `mano_v1_2/models/MANO_LEFT.pkl` and
   `MANO_RIGHT.pkl` to `DART/mano_v1_2/models/`
3. `..\bin\uv.exe pip install --python .venv\Scripts\python.exe --no-deps chumpy`
4. `python merge_smplh.py` builds the merged SMPLH_MALE/FEMALE/NEUTRAL.pkl
   in the right place

Both sites use the same MPG account system but separate registrations. The
download endpoint respects a browser session cookie:
`curl.exe -L -o <file> -H "Cookie: PHPSESSID=<from browser>" "<download url>"`

### 5. Unity rig

The DesktopFBT puppet rig lives in the VRChat avatar project, NOT this repo.
`unity_assets/Editor/GabrielFBTRigBuilder.cs` rebuilds ALL of it from
nothing: select the avatar root, Tools > ProjectGabriel > Build Desktop FBT
Rig. It generates every animation clip, the Action controller, the VRC
expression parameters and menu, the hips proxy transforms and constraint,
and wires the avatar descriptor. Then reupload the avatar.

Rig facts (mirrored in retarget.py, keep in sync):

- 29 synced 8-bit floats + 1 bool = 233/256 sync bits.
  26 muscle params, HipsY, HipsPitch, HipsRoll, FBT/Enable.
- Action layer, one Direct Blend Tree, every child weighted by FBT/One (1.0).
  Each param is a Simple1D tree over min/max clips writing humanoid muscle
  curves with chain weights (SpineFB = Spine 1.0 + Chest 0.7 + UpperChest 0.5).
- HipsY = humanoid RootT.y. mid is the avatar standing body height
  (0.98475 for gabriel), +0.80 smpl meters up (real jump apex, dart lifts
  the pelvis ~0.74m on "a person jumps up high"), -1.00 down (pelvis
  exactly on the floor at param -1). rig units = smpl meters * 1.05346
  (in-game fudge measured against the 1.0148 humanScale avatar).
- Hips rotation: pitch +-90 deg (+1 = forward), roll +-90 (+1 = his right),
  via two proxy transforms under the avatar root
  (FBT_HipsPitchProxy euler x -> FBT_HipsRollProxy euler z) and a
  VRCRotationConstraint on the Hips bone following the roll proxy. The
  constraint GlobalWeight animates to 1 inside the puppet state and
  write-defaults returns it to the scene value 0 on exit.
- WHY the constraint dance: unity nlerp-averages humanoid body rotation
  (RootQ) across ALL direct blend tree children. ~30 muscle clips each vote
  "upright" and crush 90 deg to ~3. RootQ on layers above 0 is discarded
  outright, and a full body muscle layer above stomps it back to identity.
  Generic transform curves blend per curve with zero dilution, hence the
  proxies. Do not "simplify" this back to RootQ, it cannot work.
- Puppet state behaviours: VRCAnimatorTrackingControl all body parts ->
  Animation, VRCPlayableLayerControl weight -> 1, locomotion stays ENABLED
  (that is how OSC move inputs walk the capsule while the puppet animates).
  Reset state flips everything back and auto-exits to Idle.

`muscle_ranges.json` in this folder is the dump of unity muscle limits +
humanScale + hips travel used by retarget.py. Regenerate with
Tools > ProjectGabriel > Export muscle_ranges.json if the avatar changes.

## Running

```
motion_server\.venv\Scripts\python.exe motion_server\server.py --model hml3d
```

Flags: `--model babel|hml3d`, `--port 8765`, `--guidance 5.0` (3 = looser
and more natural, 7 = obeys prompts harder but stiff), `--respacing`
(override sampling, `''` = full 10 step, `ddim5` = fast; default follows the
model), `--raw` (include raw smpl data in frames).

Main app side: `config.yml` -> `motion:` section (`enabled: true`,
`server_host`/`server_port` pointing at the GPU box). Gemini gets three
tools: `playMotion(prompt, seconds?)`, `stopMotion()`, `resetPose()`.

## Websocket protocol

Client -> server (json): `{"type": "prompt", "text": ...}` (unpauses),
`{"type": "stop"}` (back to idle prompt), `{"type": "pause"}`,
`{"type": "reset"}` (reseed rollout from the stand pose, pause, gpu idle).

Server -> client at model fps: `{"type": "frame", "t": n, "params": {...}}`
where params holds the 29 FBT values plus locomotion extras prefixed `_`:
`_vfwd`/`_vside` (m/s body frame), `_vyaw` (rad/s, + = right), `_x`/`_y`/`_yaw`.

The client maps velocities to VRChat analog inputs (`/input/Vertical`,
`/input/Horizontal`, `/input/LookHorizontal`) with a 0.3 deadzone snap
(VRChat ignores less), walk_full_speed / turn_full_rate scaling from config,
and a slower smoothing tau (0.4s) than the muscles (0.07s) because gait
velocity pulses every step.

### Closed loop locomotion (pose_tracking)

By default locomotion is open loop: DART velocities go straight to the
analog sticks and nothing checks where the avatar actually ended up, so
long walks drift (stick response, collisions, frame pacing). With the pose
exfil shader strip on the avatar (the same one voxel nav uses) the client
can close the loop. Turn it on in `config.yml` under `motion:` by setting
`pose_tracking: true` (and `pose_monitor` if VRChat is not on the primary
monitor).

How it works (`PoseTracker` in `src/motion_client.py`):
- a `PoseExfilReader` polls the screen strip at 20 Hz for the real world
  position + yaw
- when the puppet activates (and after every reset) the first good pose is
  paired with DART's `_x`/`_y`/`_yaw` as an anchor mapping DART world onto
  VRChat world. DART is z-up (ground x/y), Unity y-up (ground x/z), both
  yaws are right-positive, so the mapping is a plain 2d rotation by the
  anchor heading offset
- every frame the target pose = anchor ⊕ DART displacement, error is taken
  in the avatar's body frame and fed as a P correction ON TOP of the
  velocity feedforward: KP_POS 1.5 /s capped at 1.2 m/s, KP_YAW 2.0 /s
  capped at 1.5 rad/s
- poses older than 0.6s are ignored (decode hiccup, correction free-wheels
  on feedforward alone); blind for 3s drops the anchor; error over 5m
  (teleport/respawn) re-anchors instead of chasing

If the strip can't be read at all it behaves exactly like open loop, so
the option is safe to leave on.


## Calibration workflow (after retarget math or model changes)

1. `python probe_axes.py "<stand prompt>" 150 <model>` prints a REST dict.
   Paste into `REST_PRESETS[<model>]` in retarget.py. Stand prompts:
   babel "stand", hml3d "a person stands still without moving".
2. Measure the standing lowest joint z (min over `frame['joints'][:, 2]`)
   and set `FLOOR_DROP[<model>]` (babel 0.984, hml3d 0.971). The ground
   clamp shifts frames up so nothing sinks under the dart floor, never down.
3. `python test_generate.py "<prompt>" <seconds> <model>` sanity checks
   speed and param ranges without vrchat.
4. `python ../scripts/test_motion_stream.py <host> "<prompt>"` streams to
   vrchat from a terminal, stdin switches prompts.
5. `python ../scripts/test_hips_axes.py` sweeps HipsPitch/HipsRoll over OSC
   so you can eyeball in game which way he leans (how the RootQ bug was found).

## Retargeting conventions (retarget.py)

- DART world is z-up. SMPL local joint frames are axis aligned with the
  model frame at rest, so swing/twist decomposition against rest bone
  directions works everywhere.
- Arms are solved from PREDICTED JOINT POSITIONS (elbow-shoulder and
  wrist-elbow directions), not rotation matrices: coronal plane angle for
  ArmUp with a rest-hold blend near the pole, asin toward forward for ArmFB,
  forearm azimuth around the upper arm for twist with a blend-to-rest as the
  elbow straightens.
- NEUTRAL is the in-game verified standing pose in param space (unity muscle
  0 is NOT anatomical rest). Arm values were numerically fit so wrists land
  where darts stand puts them (0.29m lateral, ArmTW 0.4 rotates the elbow
  flexion plane outward, this is what keeps hands off the crotch).
- Mapping: raw angle -> subtract REST_RAD (per model) -> SIGN flip ->
  NEUTRAL anchor in muscle-degree space -> divide by chain gain -> clamp.
- Yaw extraction blends the pelvis forward axis with the up axis so lying
  or bowing does not gimbal the locomotion heading.

## Known limits

- Spine chain maxes at ~78 deg of flexion (unity muscle limits); pelvis
  pitch carries the rest of a deep bow, works fine in practice.
- 8-bit param sync means 0.7 deg steps on the 90 deg hips ranges for REMOTE
  viewers (local is float precision, and vrchat interpolates).
- Rollouts can drift airborne (jumps that never land). Pelvis height is
  model input, so a floating history is out of distribution and gets worse
  on its own. The engine applies gravity: if the lowest joint stays more
  than 5cm off the floor past 0.7s of continuous airtime, the whole rollout
  state (history + queued future, rigid shift so deltas stay consistent)
  sinks at 1.5 m/s until floor contact. Tunables are constants on
  MotionEngine in server.py.
- The generation quality ceiling is DART itself (2022). The researched
  upgrade path is CLoSD DiP (realtime autoregressive, hml3d trained), a
  proper porting project.
