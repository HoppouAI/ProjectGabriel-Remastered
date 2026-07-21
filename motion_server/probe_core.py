# measure core27 rest angles off a generated stand clip and print the
# CORE_REST dict for retarget_core.py, plus the stand height constants.
# usage: python probe_core.py [ardy_probe_stand.npz]
# generates a fresh stand clip with the engine if the npz doesn't exist.

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import retarget as base
import retarget_core as rc

path = Path(sys.argv[1] if len(sys.argv) > 1 else 'ardy_probe_stand.npz')
if path.exists():
    d = dict(np.load(path))
else:
    print(f'{path} not found, generating a stand clip...')
    from ardy_engine import ArdyEngine
    engine = ArdyEngine()
    engine.set_prompt('a person stands still')
    frames = [engine.next_frame() for _ in range(150)]
    d = {
        'posed_joints': np.stack([f['joints'] for f in frames]),
        'global_rot_mats': np.stack([f['rotmats'] for f in frames]),
        'global_root_heading': np.stack([f['heading'] for f in frames]),
        'root_positions': np.stack([f['root_pos'] for f in frames]),
    }
J, R, H = d['posed_joints'], d['global_rot_mats'], d['global_root_heading']
root = d['root_positions']

base.REST_RAD = {}  # measure raw, no offsets
sums = {}
skip = 20
for i in range(skip, J.shape[0]):
    ang = rc.extract_angles_core(R[i], J[i], H[i])
    for k, v in ang.items():
        if not k.startswith('_'):
            sums[k] = sums.get(k, 0.0) + v

count = J.shape[0] - skip
print(f"# mean over {count} stand frames from {path}")
print('CORE_REST = {')
for k in sorted(sums):
    mean = sums[k] / count
    print(f"    '{k}': {mean:+.4f},  # {np.degrees(mean):+.1f} deg")
print('}')
print(f"\nSTAND_HIPS_Y = {root[skip:, 1].mean():.3f}")
print(f"STAND_FLOOR_CLEAR = {J[skip:, :, 1].min():.3f}")
