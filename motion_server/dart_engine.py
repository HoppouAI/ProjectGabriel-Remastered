# DART backend: model loading + streaming rollout engine.
# importing this module bootstraps the vendored DART tree (sys.path, chdir,
# numpy aliases, posix path patch), so only import it when actually serving
# a DART model. the ardy backend lives in ardy_engine.py and needs none of
# this.

import sys
import os
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
import retarget as retarget_mod

MODELS = {
    'babel': {
        'checkpoint': './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt',
        'seed': './data/stand.pkl',
        'idle_prompt': 'stand',
        # 30fps playback cant afford full sampling (24fps gen), ddim5 keeps margin
        'respacing': 'ddim5',
    },
    # hml3d variant understands full sentence prompts, runs 20fps, smplh bodies
    'hml3d': {
        'checkpoint': './mld_denoiser/smplh_hml3d_2_8_4/checkpoint_300000.pt',
        'seed': './data/stand_20fps.pkl',
        'idle_prompt': 'a person stands still',
        # full 10 step sampling, what it was trained for, still 37fps vs 20 needed
        'respacing': '',
    },
}


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

    # gravity: pelvis height is model input, so a rollout that drifts airborne
    # is out of distribution and never comes down on its own (and later
    # prompts generate garbage from the floating history). after a real
    # jump's worth of airtime, sink the whole state until floor contact.
    CONTACT_TOL = 0.05   # lowest joint within this of the floor counts as grounded
    AIR_GRACE_S = 0.7    # allowed continuous airtime before gravity kicks in
    FALL_SPEED = 1.5     # m/s descent once it does

    def __init__(self, device='cuda', guidance=5.0, use_predicted_joints=True, respacing=None,
                 model='babel'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.guidance = guidance
        self.use_predicted_joints = use_predicted_joints
        spec = MODELS[model]
        if respacing is None:
            respacing = spec['respacing']
        self.respacing = respacing
        self.model_name = model
        self.lock = threading.Lock()

        print(f'loading models ({model})...')
        self.denoiser_args, self.denoiser_model, self.vae_args, self.vae_model = load_mld(
            spec['checkpoint'], self.device)
        self.diffusion_args = self.denoiser_args.diffusion_args
        self.diffusion_args.respacing = respacing
        self.diffusion = create_gaussian_diffusion(self.diffusion_args)

        body_type = getattr(self.vae_args.data_args, 'body_type', 'smplx')
        print('loading seed dataset...')
        self.dataset = SinglePrimitiveDataset(
            cfg_path=self.vae_args.data_args.cfg_path,
            dataset_path=self.vae_args.data_args.data_dir,
            sequence_path=spec['seed'],
            batch_size=1,
            device=self.device,
            enforce_gender='male',
            enforce_zero_beta=1,
            body_type=body_type,
        )
        self.fps = int(self.dataset.target_fps)
        self.primitive_utility = PrimitiveUtility(device=self.device, dtype=torch.float32,
                                                  body_type=body_type)
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
        self.idle_prompt = spec['idle_prompt']
        self.prompt = self.idle_prompt
        self.text_embedding = self._encode(self.idle_prompt)

        # z columns of the feature layout (transl + every joint), used by the
        # gravity shift. deltas are frame-to-frame so a rigid shift skips them.
        offsets, off = {}, 0
        for k, dim in self.primitive_utility.motion_repr.items():
            offsets[k] = off
            off += dim
        self._z_cols = [offsets['transl'] + 2] + [offsets['joints'] + 3 * j + 2 for j in range(22)]
        # seed pelvis stands at z=0, feet FLOOR_DROP below
        self.floor_z = -retarget_mod.FLOOR_DROP[model]
        self._air_time = 0.0
        print(f'engine ready on {self.device}, fps={self.fps} history={self.history_length} future={self.future_length}')

    def _encode(self, text):
        return encode_text(self.dataset.clip_model, [text], force_empty_zero=True).to(
            dtype=torch.float32, device=self.device)

    def normalize_prompt(self, text):
        """nudge whatever the client sent toward this models training style."""
        from ardy_engine import has_subject  # shared caption convention, no DART deps
        text = ' '.join(str(text).replace('.', ' ').replace('!', ' ').split())
        if not text:
            return self.idle_prompt
        low = text.lower()
        if self.model_name == 'babel':
            # babel was trained on bare action labels, strip the narration
            for pre in ('a person ', 'the person ', 'a man ', 'a woman ', 'someone '):
                if low.startswith(pre):
                    text = text[len(pre):]
                    break
        elif not has_subject(text):
            # hml3d wants humanml3d style captions, give terse verbs a subject
            text = f'a person {text}'
        return text

    def set_prompt(self, text, once=False):
        # once is an ardy backend feature (one-shot actions), dart just loops
        text = self.normalize_prompt(text)
        with self.lock:
            self.prompt = text
            self.text_embedding = self._encode(text)
            # drop queued future frames so the new prompt kicks in next primitive
            self.motion_tensor = self.motion_tensor[:, :max(self.frame_idx + 1, self.history_length), :]
        return text

    def reset(self):
        """wipe rollout context back to the seed standing pose."""
        with self.lock:
            batch = self.dataset.get_batch(batch_size=1)
            input_motions = batch[0]['motion_tensor_normalized'].to(self.device)
            motion = input_motions.squeeze(2).permute(0, 2, 1)
            self.motion_tensor = self.dataset.denormalize(motion[:, :self.history_length, :])
            self.frame_idx = 0
            self.prompt = self.idle_prompt
            self.text_embedding = self._encode(self.idle_prompt)
            self._air_time = 0.0

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
            frame = self.motion_tensor[:, self.frame_idx:self.frame_idx + 1, :]
            feat = self.primitive_utility.tensor_to_dict(frame)
            joints = feat['joints'][0, 0].reshape(22, 3)

            clearance = float(joints[:, 2].min()) - self.floor_z
            if clearance > self.CONTACT_TOL:
                self._air_time += 1.0 / self.fps
            else:
                self._air_time = 0.0
            if self._air_time > self.AIR_GRACE_S and clearance > 0.0:
                # sink history + queued future together so velocities stay
                # clean and the next rollout sees a grounded history
                dz = min(clearance, self.FALL_SPEED / self.fps)
                self.motion_tensor[..., self._z_cols] -= dz
                frame = self.motion_tensor[:, self.frame_idx:self.frame_idx + 1, :]
                feat = self.primitive_utility.tensor_to_dict(frame)
                joints = feat['joints'][0, 0].reshape(22, 3)

            transl = feat['transl'][0, 0]  # [3] world zup
            poses_6d = feat['poses_6d'][0, 0]  # [132]
            rotmats = transforms.rotation_6d_to_matrix(poses_6d.reshape(22, 6))  # [22,3,3]
            idx = self.frame_idx
            self.frame_idx += 1
        return {
            't': idx,
            'transl': transl.cpu().numpy(),
            'rotmats': rotmats.cpu().numpy(),
            'joints': joints.cpu().numpy(),
        }
