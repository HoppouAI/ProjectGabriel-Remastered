# offline smoke test for the ARDY backend: stream a prompt, retarget every
# frame, dump params to test_frames.json for the unity sampler.
# usage: python test_generate_ardy.py "a person waves" [seconds]

import sys
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from ardy_engine import ArdyEngine  # noqa: E402
from retarget_core import CoreRetargeter  # noqa: E402


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'a person walks forward'
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

    engine = ArdyEngine()
    n_frames = int(seconds * engine.fps)
    retargeter = CoreRetargeter(HERE / 'muscle_ranges.json', fps=engine.fps)
    engine.set_prompt(prompt)

    frames = []
    heights = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        f = engine.next_frame()
        params = retargeter.frame_to_params(
            f['joints'], f['rotmats'], f['heading'], f['root_pos'], f['smooth_root'])
        frames.append({'t': f['t'], 'params': {k: round(float(v), 4) for k, v in params.items()}})
        heights.append(float(f['root_pos'][1]))
    dt = time.perf_counter() - t0
    print(f'{n_frames} frames in {dt:.2f}s -> {n_frames / dt:.1f} fps overall (need {engine.fps})')
    print(f'hips y: min {min(heights):.3f} max {max(heights):.3f} mean {sum(heights)/len(heights):.3f}')

    out = HERE / 'test_frames.json'
    with open(out, 'w') as fp:
        json.dump({'prompt': prompt, 'fps': engine.fps, 'frames': frames}, fp, indent=1)
    print(f'wrote {out}')

    keys = [k for k in frames[0]['params'] if not k.startswith('_')]
    print(f'{"param":10s} {"min":>7s} {"max":>7s}')
    for k in keys:
        vals = [f['params'][k] for f in frames]
        print(f'{k:10s} {min(vals):+7.2f} {max(vals):+7.2f}')


if __name__ == '__main__':
    main()
