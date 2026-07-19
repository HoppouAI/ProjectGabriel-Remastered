# axis convention probe: print world facing/up and thigh dirs for a few frames
import sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from server import MotionEngine
import numpy as np

engine = MotionEngine()
engine.set_prompt('walk forward')

UP = np.array([0, 1, 0.0])
FWD = np.array([0, 0, 1.0])
LEFT = np.array([1, 0, 0.0])

print(f'{"t":>4s} {"face(R0@FWD)":>24s} {"up(R0@UP)":>24s} {"Lthigh dir":>24s} {"pelvis":>24s}')
for i in range(120):
    f = engine.next_frame()
    if i % 20 != 0:
        continue
    R = f['rotmats']
    face = R[0] @ FWD
    up = R[0] @ UP
    lthigh = R[0] @ (R[1] @ -UP)  # world dir of left thigh bone
    pel = f['joints'][0]
    fmt = lambda v: '[' + ' '.join(f'{x:+.2f}' for x in v) + ']'
    print(f'{f["t"]:>4d} {fmt(face):>24s} {fmt(up):>24s} {fmt(lthigh):>24s} {fmt(pel):>24s}')
