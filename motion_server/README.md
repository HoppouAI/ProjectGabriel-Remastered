# Motion Server

Real time full body motion generation for Gabriel. Type a text prompt, a
diffusion model generates human motion frame by frame, and the frames stream
over a websocket into VRChat as avatar puppet parameters. He dances, sits,
bows, lies down, waves, and walking motions physically move him through the
world.

Two backends share one server and one protocol:

- [DART](https://github.com/zkf1997/DART) (`babel`, `hml3d`): the original
  backend, vendored tree in `DART/`, runs in `.venv`
- [ARDY](https://github.com/nv-tlabs/ardy) (`core8`): nvidia's autoregressive
  motion diffusion, noticeably higher quality and better style control,
  clone in `ARDY/`, runs in `.venv-ardy`

This README documents the ENTIRE system, including the parts that live
outside this repo (the Unity rig), so everything can be rebuilt from nothing.

## Architecture

```
prompt ("a person dances energetically")
   |
   v
rollout engine (dart_engine.py or ardy_engine.py, GPU)
   |   world space body frames at model fps
   v
retarget.py / retarget_core.py               world frames -> 29 FBT
   |   muscle params + locomotion velocities  params + body frame velocity
   v
websocket (port 8765, json frames, server.py)
   |
   v
src/motion_client.py (main app)               60hz OSC sender, one euro
   |   smoothing, locomotion mapping          filter per param
   v
VRChat OSC (/avatar/parameters/FBT/*, /input/Vertical ...)
   |
   v
DesktopFBT puppet rig on the avatar           direct blend tree + VRC
                                              rotation constraint
```

Three models, selectable with `--model`:

| | `babel` (default) | `hml3d` | `core8` (ARDY) |
|---|---|---|---|
| prompts | bare action verbs ("dance") | full sentences | full sentences, best style following |
| fps | 30 | 20 | 20 |
| bodies | SMPL-X | SMPL-H | Core-27 skeleton |
| sampling | ddim5 | full 10 step | 8 of 10 steps (realtime margin on a 3060) |
| venv | `.venv` | `.venv` | `.venv-ardy` |
| quality | okay | better | best, understands zombie/limp/sneak style prompts |

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

### 6. ARDY backend (optional, the good one)

Separate venv because it wants modern torch/transformers that DART's
vendored 2022 code can't share. From `motion_server/`:

```
..\bin\uv.exe venv .venv-ardy --python 3.11
git clone https://github.com/nv-tlabs/ardy ARDY
..\bin\uv.exe pip install --python .venv-ardy\Scripts\python.exe -e ARDY
..\bin\uv.exe pip install --python .venv-ardy\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
..\bin\uv.exe pip install --python .venv-ardy\Scripts\python.exe bitsandbytes websockets
```

ARDY's foot skate cleanup is a small C++ extension (`motion_correction`),
build it with MinGW if the editable install didn't
(`python -c "from motion_correction import motion_postprocess"` to check).

The `core8` checkpoint (nvidia/ARDY-Core-RP-20FPS-Horizon8, 765MB,
Apache/NVIDIA Open Model) downloads itself into the HF cache on first run.

The text encoder is the gated part upstream: ARDY wants LLM2Vec on top of
meta-llama/Meta-Llama-3-8B-Instruct (16GB, meta approval wall). We skip the
wall with a community NF4 quant of the already-merged encoder:

```
huggingface-cli download Aero-Ex/KIMODO-Meta3_llm2vec_NF4 --local-dir text_encoders/llm2vec_nf4
```

5.5GB on disk, 4.65GB VRAM, loads through ARDY's own `LLM2VecEncoder` with
`peft_model_name_or_path=None` (the adapters are already merged in). Total
server VRAM ~5.7GB. If that repo ever disappears, the merge can be rebuilt
from the kuotient Llama-3-8B-Instruct mirror + McGill-NLP llm2vec adapters
(mntp then supervised): merge both LoRAs, quantize NF4, and mind the key
prefixes, the adapters were saved on a bare LlamaModel so their keys sit one
level shallower than the merged model expects, and mntp vs supervised nest
differently. Keep `_name_or_path` pointing at the meta repo name in
config.json so ARDY applies the right prompt wrapping.

Speed on an RTX 3060 12GB (Gate A, 6s clips): 8 denoise steps = 27fps mean /
24fps worst chunk, 10 steps = 22fps mean / 18fps worst. Default is 8 steps
(`--steps`), which holds 20fps realtime with margin.

## Running

```
motion_server\.venv\Scripts\python.exe motion_server\server.py --model hml3d
motion_server\.venv-ardy\Scripts\python.exe motion_server\server.py --model core8
```

Flags: `--model babel|hml3d|core8`, `--port 8765`, `--raw` (include raw
joint data in frames). DART only: `--guidance 5.0` (3 = looser and more
natural, 7 = obeys prompts harder but stiff), `--respacing` (override
sampling, `''` = full 10 step, `ddim5` = fast). ARDY only: `--steps 8`
(denoise steps, 10 max).

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

DART models:

1. `python probe_axes.py "<stand prompt>" 150 <model>` prints a REST dict.
   Paste into `REST_PRESETS[<model>]` in retarget.py. Stand prompts:
   babel "stand", hml3d "a person stands still without moving".
2. Measure the standing lowest joint z (min over `frame['joints'][:, 2]`)
   and set `FLOOR_DROP[<model>]` (babel 0.984, hml3d 0.971). The ground
   clamp shifts frames up so nothing sinks under the dart floor, never down.
3. `python test_generate.py "<prompt>" <seconds> <model>` sanity checks
   speed and param ranges without vrchat.

ARDY (`.venv-ardy` python):

1. `python probe_core.py [stand clip .npz]` prints CORE_REST plus the stand
   height constants, paste into retarget_core.py.
2. `python test_generate_ardy.py "<prompt>" <seconds>` sanity checks speed
   and param ranges, writes test_frames.json for the unity sampler.

Either backend:

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

### ARDY conventions (retarget_core.py)

- ARDY world is y-up meters, floor exactly y=0, faces +z at heading 0,
  +x is the model's LEFT. `global_root_heading = (cos t, sin t)` with
  positive t turning LEFT. Conversion to the DART/client convention the
  protocol speaks: `x = -x`, `y = z`, `yaw = -atan2(h1, h0)`.
- Core-27 skeleton (`cskel27`): hips/spine/neck/head chain, 6-joint arms
  (shoulder/arm/forearm/hand/handEnd/thumb), 4-joint legs. Torso and foot
  bind orientations are world aligned at stand so the anatomical basis is
  the model frame itself; arm binds are NOT world aligned, so arms, wrists
  and legs are all solved from joint positions (same formulas as the SMPL
  path), with the constant bind offsets absorbed by CORE_REST calibration.
- Yaw comes from the model's own smoothed heading channel instead of the
  pelvis axes, which is more stable when lying or tumbling.
- The engine keeps a normalized rollout tensor and feeds its tail back as
  history each `autoregressive_step`. History is deliberately SHORT: on a
  prompt switch it is cut to one 4-frame token, because a long history is an
  attractor (2s of seated history and no prompt ever gets him up again,
  nvidia's demo defaults to 4 frames for the same reason).
- Stop requests play 'a person stands up' first and only settle into the
  'stands still' idle once the pose is actually upright (hips above 0.75m,
  pelvis up axis vertical, 10s hard cap). 'stands still' describes a state,
  not a transition, so from sitting/lying it would just hold the pose.

## Known limits

- Spine chain maxes at ~78 deg of flexion (unity muscle limits); pelvis
  pitch carries the rest of a deep bow, works fine in practice.
- 8-bit param sync means 0.7 deg steps on the 90 deg hips ranges for REMOTE
  viewers (local is float precision, and vrchat interpolates).
- Rollouts can drift airborne (jumps that never land). Pelvis height is
  model input, so a floating history is out of distribution and gets worse
  on its own. The DART engine applies gravity: if the lowest joint stays
  more than 5cm off the floor past 0.7s of continuous airtime, the whole
  rollout state (history + queued future, rigid shift so deltas stay
  consistent) sinks at 1.5 m/s until floor contact. Tunables are constants
  on MotionEngine in dart_engine.py. ARDY hasn't needed this so far (its
  training data grounds much harder); if it starts floating, port the same
  trick into ardy_engine.py.
- ARDY generates in 0.4s chunks on a lookahead worker thread: the next chunk
  renders on the gpu while the current one streams, so frames leave evenly at
  20fps (measured p50 gap 47ms) instead of stall-then-burst. The server also
  carries up to 0.6s of catch-up debt for chunks that overrun. If it still
  hiccups on a shared gpu (vrchat on the same machine), drop to `--steps 6`.
- Android/Quest VRChat does not apply animations to VRC constraint
  properties, so the hips rotation constraint can't be engaged by the puppet
  state there. Quest builds of the rig run the constraint always-on instead
  (GlobalWeight 1 in the scene, set automatically by the rig builder when the
  active build target is Android). Cost: hips don't sway during normal
  locomotion on quest. Both platform uploads must be redone whenever the
  pitch/roll/HipsY ranges change, each carries its own baked clips.
