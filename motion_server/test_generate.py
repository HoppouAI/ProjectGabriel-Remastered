# offline smoke test: load the engine, roll out a few seconds of motion for
# a prompt, run every frame through the retargeter, dump params to json for
# the unity sampler to verify. no websocket involved.
# usage: python test_generate.py "walk forward" [seconds]

import sys
import json
import time
from pathlib import Path

HERE = Path(__file__).parent

from dart_engine import MotionEngine  # noqa: E402  (dart_engine sets up sys.path/chdir)
from retarget import Retargeter  # noqa: E402


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'walk forward'
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    model = sys.argv[3] if len(sys.argv) > 3 else 'babel'

    engine = MotionEngine(model=model)
    n_frames = int(seconds * engine.fps)
    retargeter = Retargeter(HERE / 'muscle_ranges.json', fps=engine.fps)
    engine.set_prompt(prompt)

    frames = []
    t0 = time.perf_counter()
    t_warm = None
    pelvis_z = []
    for i in range(n_frames):
        f = engine.next_frame()
        params = retargeter.frame_to_params(f['transl'], f['rotmats'], f['joints'])
        frames.append({'t': f['t'], 'params': {k: round(float(v), 4) for k, v in params.items()}})
        pelvis_z.append(float(f['joints'][0][2]))
        if i == 29:
            t_warm = time.perf_counter()
    dt = time.perf_counter() - t0
    print(f'{n_frames} frames in {dt:.2f}s -> {n_frames / dt:.1f} fps overall')
    if t_warm and n_frames > 30:
        steady = (n_frames - 30) / (time.perf_counter() - t_warm)
        print(f'steady state (after 30 frame warmup): {steady:.1f} fps')
    print(f'pelvis z: min {min(pelvis_z):.3f} max {max(pelvis_z):.3f} mean {sum(pelvis_z)/len(pelvis_z):.3f}')

    out = HERE / 'test_frames.json'
    with open(out, 'w') as fp:
        json.dump({'prompt': prompt, 'fps': engine.fps, 'frames': frames}, fp, indent=1)
    print(f'wrote {out}')

    # quick eyeball: param ranges across the clip
    keys = [k for k in frames[0]['params'] if not k.startswith('_')]
    print(f'{"param":10s} {"min":>7s} {"max":>7s}')
    for k in keys:
        vals = [f['params'][k] for f in frames]
        print(f'{k:10s} {min(vals):+7.2f} {max(vals):+7.2f}')


if __name__ == '__main__':
    main()
