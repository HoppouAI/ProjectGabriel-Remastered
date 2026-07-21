# diagnostics: measure the raw anatomical rest angles of a models 'stand'
# prompt and print a REST dict ready to paste into retarget.py REST_PRESETS.
# usage: python probe_axes.py [prompt] [frames] [model]

import sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from dart_engine import MotionEngine
from retarget import extract_angles
import numpy as np

prompt = sys.argv[1] if len(sys.argv) > 1 else 'stand'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
model = sys.argv[3] if len(sys.argv) > 3 else 'babel'

engine = MotionEngine(model=model)
engine.set_prompt(prompt)

sums = {}
zs = []
for i in range(n):
    f = engine.next_frame()
    if i < 30:  # skip transition from seed
        continue
    ang = extract_angles(f['rotmats'], f['joints'])
    for k, v in ang.items():
        sums[k] = sums.get(k, 0.0) + v
    zs.append(float(f['joints'][0][2]))

count = n - 30
print(f"\n# mean over {count} '{prompt}' frames, pelvis z mean {sum(zs)/len(zs):+.3f}")
print('REST_RAD = {')
for k in sorted(sums):
    if k.startswith('_'):
        continue
    mean = sums[k] / count
    print(f"    '{k}': {mean:+.4f},  # {np.degrees(mean):+.1f} deg")
print('}')
