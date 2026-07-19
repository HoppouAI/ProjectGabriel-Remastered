# DART motion streaming server
# runs on the GPU box, generates motion from text prompts and streams
# world-space SMPL-X frames (plus retargeted FBT params) over websocket.
#
# usage: python server.py [--host 0.0.0.0] [--port 8765]
# needs: DART/ checkpoint tree + SMPL-X body models (see README)

import sys
import os
import json
import time
import argparse
import asyncio
import threading
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'shims'))
sys.path.insert(0, str(HERE / 'DART'))
os.chdir(HERE / 'DART')

# numpy>=1.24 removed these, DART's vendored humanml code still uses them
import numpy as np
for _alias, _t in [('float', float), ('int', int), ('bool', bool), ('object', object), ('str', str)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _t)

# checkpoints and args.yaml were written on linux and contain PosixPath objects
import pathlib
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

import torch
import yaml
import tyro
from dataclasses import asdict

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from pytorch3d import transforms
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
from utils.smpl_utils import PrimitiveUtility
from utils.misc_util import encode_text
from mld.train_mvae import Args as MVAEArgs
from mld.train_mld import DenoiserArgs, MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs

sys.path.insert(0, str(HERE))
from retarget import Retargeter

DENOISER_CHECKPOINT = './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
FPS = 30
DEFAULT_PROMPT = 'stand'


class ClassifierFreeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, y=None):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y['uncond'] = True
        out_uncond = self.model(x, timesteps, y)
        return out_uncond + (y['scale'] * (out - out_uncond))


def load_mld(denoiser_checkpoint, device):
    denoiser_dir = Path(denoiser_checkpoint).parent
    with open(denoiser_dir / 'args.yaml', 'r') as f:
        denoiser_args = tyro.extras.from_yaml(MLDArgs, yaml.safe_load(f)).denoiser_args

    denoiser_class = DenoiserMLP if isinstance(denoiser_args.model_args, DenoiserMLPArgs) else DenoiserTransformer
    denoiser_model = denoiser_class(**asdict(denoiser_args.model_args)).to(device)
    checkpoint = torch.load(denoiser_checkpoint, map_location=device, weights_only=False)
    denoiser_model.load_state_dict(checkpoint['model_state_dict'])
    for p in denoiser_model.parameters():
        p.requires_grad = False
    denoiser_model.eval()
    denoiser_model = ClassifierFreeWrapper(denoiser_model)

    vae_dir = Path(denoiser_args.mvae_path).parent
    with open(vae_dir / 'args.yaml', 'r') as f:
        vae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(f))
    vae_model = AutoMldVae(**asdict(vae_args.model_args)).to(device)
    checkpoint = torch.load(denoiser_args.mvae_path, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict']
    if 'latent_mean' not in state:
        state['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in state:
        state['latent_std'] = torch.tensor(1)
    vae_model.load_state_dict(state)
    vae_model.latent_mean = state['latent_mean']
    vae_model.latent_std = state['latent_std']
    for p in vae_model.parameters():
        p.requires_grad = False
    vae_model.eval()

    return denoiser_args, denoiser_model, vae_args, vae_model


class MotionEngine:
    """owns the DART rollout state. all methods must be called from one thread."""

    def __init__(self, device='cuda', guidance=5.0, use_predicted_joints=True, respacing='ddim5'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.guidance = guidance
        self.use_predicted_joints = use_predicted_joints
        self.respacing = respacing
        self.lock = threading.Lock()

        print('loading models...')
        self.denoiser_args, self.denoiser_model, self.vae_args, self.vae_model = load_mld(
            DENOISER_CHECKPOINT, self.device)
        self.diffusion_args = self.denoiser_args.diffusion_args
        self.diffusion_args.respacing = respacing
        self.diffusion = create_gaussian_diffusion(self.diffusion_args)

        print('loading seed dataset...')
        self.dataset = SinglePrimitiveDataset(
            cfg_path=self.vae_args.data_args.cfg_path,
            dataset_path=self.vae_args.data_args.data_dir,
            sequence_path='./data/stand.pkl',
            batch_size=1,
            device=self.device,
            enforce_gender='male',
            enforce_zero_beta=1,
        )
        self.primitive_utility = PrimitiveUtility(device=self.device, dtype=torch.float32)
        self.history_length = self.dataset.history_length
        self.future_length = self.dataset.future_length
        self.primitive_length = self.history_length + self.future_length

        batch = self.dataset.get_batch(batch_size=1)
        input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
        del model_kwargs['y']['motion_tensor_normalized']
        self.gender = model_kwargs['y']['gender'][0]
        self.betas = model_kwargs['y']['betas'][:, :self.primitive_length, :].to(self.device)
        self.pelvis_delta = self.primitive_utility.calc_calibrate_offset({
            'betas': self.betas[:, 0, :],
            'gender': self.gender,
        })
        input_motions = input_motions.to(self.device)
        motion = input_motions.squeeze(2).permute(0, 2, 1)  # [1, T, D]
        self.motion_tensor = self.dataset.denormalize(motion[:, :self.history_length, :])
        self.frame_idx = 0
        self.prompt = DEFAULT_PROMPT
        self.text_embedding = self._encode(DEFAULT_PROMPT)
        print(f'engine ready on {self.device}, history={self.history_length} future={self.future_length}')

    def _encode(self, text):
        return encode_text(self.dataset.clip_model, [text], force_empty_zero=True).to(
            dtype=torch.float32, device=self.device)

    def set_prompt(self, text):
        with self.lock:
            self.prompt = text
            self.text_embedding = self._encode(text)
            # drop queued future frames so the new prompt kicks in next primitive
            self.motion_tensor = self.motion_tensor[:, :max(self.frame_idx + 1, self.history_length), :]

    @torch.no_grad()
    def _rollout(self):
        sample_fn = self.diffusion.ddim_sample_loop if self.respacing else self.diffusion.p_sample_loop
        guidance_param = torch.ones(1, *self.denoiser_args.model_args.noise_shape,
                                    device=self.device) * self.guidance
        history_motion = self.motion_tensor[:, -self.history_length:, :]

        history_feature_dict = self.primitive_utility.tensor_to_dict(history_motion)
        transf_rotmat = torch.eye(3, device=self.device).unsqueeze(0)
        transf_transl = torch.zeros(1, 1, 3, device=self.device)
        history_feature_dict.update({
            'transf_rotmat': transf_rotmat,
            'transf_transl': transf_transl,
            'gender': self.gender,
            'betas': self.betas[:, :self.history_length, :],
            'pelvis_delta': self.pelvis_delta,
        })
        canonicalized_history, blended = self.primitive_utility.get_blended_feature(
            history_feature_dict, use_predicted_joints=self.use_predicted_joints)
        transf_rotmat = canonicalized_history['transf_rotmat']
        transf_transl = canonicalized_history['transf_transl']
        history_motion_normalized = self.dataset.normalize(self.primitive_utility.dict_to_tensor(blended))

        y = {
            'text_embedding': self.text_embedding,
            'history_motion_normalized': history_motion_normalized,
            'scale': guidance_param,
        }
        x_start = sample_fn(
            self.denoiser_model,
            (1, *self.denoiser_args.model_args.noise_shape),
            clip_denoised=False,
            model_kwargs={'y': y},
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        latent = x_start.permute(1, 0, 2)
        future = self.vae_model.decode(latent, history_motion_normalized,
                                       nfuture=self.future_length,
                                       scale_latent=self.denoiser_args.rescale_latent)
        future = self.dataset.denormalize(future)
        future_dict = self.primitive_utility.tensor_to_dict(future)
        future_dict.update({
            'transf_rotmat': transf_rotmat,
            'transf_transl': transf_transl,
            'gender': self.gender,
            'betas': self.betas[:, :self.future_length, :],
            'pelvis_delta': self.pelvis_delta,
        })
        future_dict = self.primitive_utility.transform_feature_to_world(future_dict)
        future_tensor = self.primitive_utility.dict_to_tensor(future_dict)
        self.motion_tensor = torch.cat([self.motion_tensor, future_tensor], dim=1)

    @torch.no_grad()
    def next_frame(self):
        """generate as needed and return the next frame as plain python data."""
        with self.lock:
            while self.frame_idx >= self.motion_tensor.shape[1]:
                self._rollout()
            feat = self.primitive_utility.tensor_to_dict(
                self.motion_tensor[:, self.frame_idx:self.frame_idx + 1, :])
            transl = feat['transl'][0, 0]  # [3] world zup
            poses_6d = feat['poses_6d'][0, 0]  # [132]
            joints = feat['joints'][0, 0].reshape(22, 3)  # world zup
            rotmats = transforms.rotation_6d_to_matrix(poses_6d.reshape(22, 6))  # [22,3,3]
            idx = self.frame_idx
            self.frame_idx += 1
        return {
            't': idx,
            'transl': transl.cpu().numpy(),
            'rotmats': rotmats.cpu().numpy(),
            'joints': joints.cpu().numpy(),
        }


async def client_loop(ws, engine, retargeter, send_raw):
    print(f'client connected: {ws.remote_address}')

    async def receiver():
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get('type') == 'prompt':
                text = str(msg.get('text', '')).strip() or DEFAULT_PROMPT
                print(f'prompt: {text!r}')
                await asyncio.to_thread(engine.set_prompt, text)
            elif msg.get('type') == 'stop':
                print('prompt: stop -> stand')
                await asyncio.to_thread(engine.set_prompt, DEFAULT_PROMPT)

    recv_task = asyncio.create_task(receiver())
    frame_interval = 1.0 / FPS
    next_send = time.monotonic()
    try:
        while not recv_task.done():
            frame = await asyncio.to_thread(engine.next_frame)
            payload = {'type': 'frame', 't': frame['t']}
            if retargeter is not None:
                payload['params'] = retargeter.frame_to_params(
                    frame['transl'], frame['rotmats'], frame['joints'])
            if send_raw:
                payload['smplx'] = {
                    'transl': frame['transl'].tolist(),
                    'rotmats': frame['rotmats'].reshape(-1).tolist(),
                    'joints': frame['joints'].reshape(-1).tolist(),
                }
            await ws.send(json.dumps(payload))
            next_send += frame_interval
            delay = next_send - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_send = time.monotonic()  # fell behind, resync
    except Exception as e:
        print(f'client loop ended: {e}')
    finally:
        recv_task.cancel()
        print('client disconnected')


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--respacing', default='ddim5', help="'' for full 10 step sampling")
    ap.add_argument('--raw', action='store_true', help='include raw smplx data in frames')
    args = ap.parse_args()

    engine = MotionEngine(guidance=args.guidance, respacing=args.respacing)

    ranges_path = HERE / 'muscle_ranges.json'
    retargeter = None
    if ranges_path.exists():
        retargeter = Retargeter(ranges_path)
        print('retargeter loaded')
    else:
        print('muscle_ranges.json not found, streaming raw smplx only')

    import websockets
    async def handler(ws):
        await client_loop(ws, engine, retargeter, send_raw=args.raw or retargeter is None)

    async with websockets.serve(handler, args.host, args.port, max_size=None):
        print(f'motion server listening on ws://{args.host}:{args.port}')
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
