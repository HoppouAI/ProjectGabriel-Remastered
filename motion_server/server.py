# motion streaming server
# runs on the GPU box, generates motion from text prompts and streams
# retargeted FBT params over websocket. two backends:
#   dart (babel/hml3d): vendored DART tree, smpl bodies (dart_engine.py)
#   ardy (core8): nvidia ARDY autoregressive diffusion (ardy_engine.py)
#
# usage: python server.py [--model hml3d|babel|core8] [--host 0.0.0.0] [--port 8765]

import json
import time
import argparse
import asyncio
from pathlib import Path

HERE = Path(__file__).parent

DART_MODELS = ('babel', 'hml3d')
ARDY_MODELS = ('core8',)


def build_backend(args):
    """load the requested engine + retargeter. returns (engine, retargeter,
    params_of, raw_of)."""
    ranges_path = HERE / 'muscle_ranges.json'
    if not ranges_path.exists():
        raise SystemExit('muscle_ranges.json not found, dump it from unity first')

    if args.model in ARDY_MODELS:
        from ardy_engine import ArdyEngine
        from retarget_core import CoreRetargeter
        engine = ArdyEngine(model=args.model, steps=args.steps, hist_cap_s=args.history)
        retargeter = CoreRetargeter(ranges_path, fps=engine.fps)
        print(f'retargeter loaded (rest preset: core, {engine.fps}fps)')

        def params_of(f):
            return retargeter.frame_to_params(
                f['joints'], f['rotmats'], f['heading'], f['root_pos'], f['smooth_root'])

        def raw_of(f):
            return {'joints': f['joints'].reshape(-1).tolist()}

        return engine, retargeter, params_of, raw_of

    # importing dart_engine bootstraps the vendored DART tree (chdir etc)
    import dart_engine
    import retarget as retarget_mod
    from retarget import Retargeter
    engine = dart_engine.MotionEngine(guidance=args.guidance, respacing=args.respacing,
                                      model=args.model)
    retarget_mod.set_rest(args.model)
    retargeter = Retargeter(ranges_path, fps=engine.fps)
    print(f'retargeter loaded (rest preset: {args.model}, {engine.fps}fps)')

    def params_of(f):
        return retargeter.frame_to_params(f['transl'], f['rotmats'], f['joints'])

    def raw_of(f):
        return {
            'transl': f['transl'].tolist(),
            'rotmats': f['rotmats'].reshape(-1).tolist(),
            'joints': f['joints'].reshape(-1).tolist(),
        }

    return engine, retargeter, params_of, raw_of


async def client_loop(ws, engine, retargeter, params_of, raw_of, send_raw):
    print(f'client connected: {ws.remote_address}')
    state = {'paused': False}

    async def receiver():
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get('type')
            if mtype == 'prompt':
                text = str(msg.get('text', '')).strip()
                used = await asyncio.to_thread(engine.set_prompt, text)
                print(f'prompt: {text!r} -> {used!r}')
                state['paused'] = False
            elif mtype == 'stop':
                print('prompt: stop -> idle')
                await asyncio.to_thread(engine.set_prompt, engine.idle_prompt)
            elif mtype == 'pause':
                print('paused')
                state['paused'] = True
            elif mtype == 'reset':
                print('reset -> idle stand, paused')
                await asyncio.to_thread(engine.reset)
                retargeter.reset_root()
                state['paused'] = True

    recv_task = asyncio.create_task(receiver())
    frame_interval = 1.0 / engine.fps
    max_backlog = 0.6  # seconds of catch-up credit after a generation stall
    next_send = time.monotonic()
    try:
        while not recv_task.done():
            if state['paused']:
                await asyncio.sleep(0.05)
                next_send = time.monotonic()
                continue
            frame = await asyncio.to_thread(engine.next_frame)
            payload = {'type': 'frame', 't': frame['t'], 'params': params_of(frame)}
            if send_raw:
                payload['smplx'] = raw_of(frame)
            await ws.send(json.dumps(payload))
            next_send += frame_interval
            delay = next_send - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            elif -delay > max_backlog:
                # a chunk took way too long, drop the excess debt so we dont
                # fast-forward more than the client can absorb
                next_send = time.monotonic() - max_backlog
    except Exception as e:
        print(f'client loop ended: {e}')
    finally:
        recv_task.cancel()
        print('client disconnected')


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--model', default='babel', choices=list(DART_MODELS + ARDY_MODELS),
                    help='babel = verb prompts 30fps, hml3d = sentence prompts 20fps, '
                         'core8 = ardy sentence prompts 20fps (needs .venv-ardy)')
    ap.add_argument('--guidance', type=float, default=5.0, help='dart classifier-free guidance')
    ap.add_argument('--respacing', default=None,
                    help="dart sampling override: '' = full 10 step, 'ddim5' fast. default follows the model")
    ap.add_argument('--steps', type=int, default=8, help='ardy denoising steps (10 max, 8 keeps realtime margin)')
    ap.add_argument('--history', type=float, default=2.0,
                    help='ardy context seconds fed back each step. longer = steadier long '
                         'holds (sitting still), shorter = snappier but drifts and freaks '
                         'out over time. prompt switches always cut to one token regardless')
    ap.add_argument('--raw', action='store_true', help='include raw joint data in frames')
    args = ap.parse_args()

    engine, retargeter, params_of, raw_of = build_backend(args)

    import websockets

    async def handler(ws):
        await client_loop(ws, engine, retargeter, params_of, raw_of, send_raw=args.raw)

    async with websockets.serve(handler, args.host, args.port, max_size=None):
        print(f'motion server listening on ws://{args.host}:{args.port} (model {args.model})')
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
