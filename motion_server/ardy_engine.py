# ARDY streaming engine: same duck-typed surface as dart_engine.MotionEngine
# (fps / idle_prompt / normalize_prompt / set_prompt / reset / next_frame)
# but backed by nvidia ARDY core checkpoints instead of DART.
#
# streaming pattern mirrors the official interactive demo's _generate_step:
# keep a normalized motion tensor, feed its tail (a multiple of the token
# size) as history each autoregressive_step, append the new horizon, decode
# with motion_rep.inverse. text encoder is the community NF4 quant of the
# gated LLM2Vec llama so no meta approval is needed (~5.5GB vram, encodes a
# prompt in ~0.3s, cached).

import threading
from pathlib import Path

import numpy as np
import torch

ENCODER_PATH = str(Path(__file__).parent / 'text_encoders' / 'llm2vec_nf4')


class ArdyEngine:
    """owns the ARDY rollout state. all methods must be called from one thread."""

    def __init__(self, model='core8', device='cuda', steps=8, hist_cap_s=2.0,
                 cfg_text=2.0, cfg_constraint=2.0):
        # hist_cap_s is the STEADY STATE context fed back each step. long =
        # stable poses (less drift during long holds), short = the model
        # forgets where it is and starts drifting/freaking after ~30s of
        # sitting still. prompt switches always cut to one token regardless,
        # thats what keeps poses from becoming attractors, so this knob only
        # trades stability vs a bit of gpu time.
        from ardy.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
        from ardy.model.load_model import load_model

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()

        print(f'loading text encoder ({ENCODER_PATH})...')
        self._encoder = LLM2VecEncoder(
            base_model_name_or_path=ENCODER_PATH,
            peft_model_name_or_path=None,
            dtype='float16', llm_dim=4096, device=str(self.device),
        )
        print(f'loading ardy ({model})...')
        self.model = load_model(model, device=str(self.device), text_encoder=self._encoder)
        self.model_name = model

        self.fps = int(self.model.motion_rep.fps)
        self._patch = int(self.model.num_frames_per_token)
        self._horizon = int(self.model.gen_horizon_len)
        base_steps = int(getattr(getattr(self.model, 'diffusion', None), 'num_base_steps', 10) or 10)
        self._steps = min(int(steps), base_steps)
        self._cfg = (float(cfg_text), float(cfg_constraint))
        self._hist_cap = max(self._patch, int(hist_cap_s * self.fps) // self._patch * self._patch)

        self.idle_prompt = 'a person stands still'
        # 'stands still' describes a state, not a transition: from a seated
        # or lying history the model just keeps holding the pose. going to
        # idle therefore plays a get-up prompt first (breaks the attractor in
        # ~2.5s measured), then settles into the real idle.
        self.recover_prompt = 'a person stands up'
        self._recover_s = 3.0
        self._recover_max_s = 10.0
        self._recover_at = None   # earliest absolute frame to swap recover -> idle
        self._recover_cap = None  # hard cap, settle even if not upright
        # one-shot actions (backflip etc): instead of looping the prompt
        # forever, watch for the body to leave the standing pose and come
        # back to it, then settle into idle. a timer cap catches actions
        # that never return upright (or never leave it, like waving).
        self._once_active = False
        self._once_left = False
        self._once_run = 0
        self._once_deadline = 0
        self._once_max_s = 8.0
        self._once_settle_s = 0.5
        self._emb_cache = {}

        self._motion = None        # [1, T, D] normalized, tail of the rollout
        self._motion_base = 0      # absolute index of _motion[:, 0]
        self._frames = []          # decoded per-frame dicts, parallel tail
        self._frames_base = 0
        self.frame_idx = 0         # absolute index of the next frame to serve
        self.prompt = self.idle_prompt
        self._text_feat = self._encode(self.idle_prompt)

        # lookahead worker: generates the next chunk on the gpu while the
        # current one streams out, otherwise every chunk boundary is a ~300ms
        # stall followed by an 8 frame burst and motion looks like a slideshow
        self._cond = threading.Condition(self.lock)
        self._epoch = 0            # bumped on prompt switch / reset, voids in-flight chunks
        self._lookahead = 2 * self._horizon
        self._worker_err = None
        threading.Thread(target=self._worker_loop, daemon=True).start()
        print(f'ardy engine ready on {self.device}, fps={self.fps} patch={self._patch} '
              f'horizon={self._horizon} steps={self._steps}')

    # -- prompt handling --

    def _encode(self, text):
        feat = self._emb_cache.get(text)
        if feat is None:
            emb, _ = self._encoder([text])
            feat = (emb if emb.dim() == 3 else emb[None]).to(self.device)
            if len(self._emb_cache) > 64:
                self._emb_cache.clear()
            self._emb_cache[text] = feat
        return feat

    def normalize_prompt(self, text):
        """ardy wants humanml style captions, same as the hml3d dart variant."""
        text = ' '.join(str(text).replace('.', ' ').replace('!', ' ').split())
        if not text:
            return self.idle_prompt
        if 'person' not in text.lower():
            text = f'a person {text}'
        return text

    def _truncate_to_served(self, cut_history):
        """drop frames generated under the previous prompt. with cut_history,
        also shrink kept history to one token so the old pose stops acting as
        an attractor (long history dominates the text condition)."""
        self._epoch += 1  # void any chunk the worker has in flight
        self._cond.notify_all()
        if self._motion is None:
            return
        keep = max(self.frame_idx + 1 - self._motion_base, self._patch)
        keep = min(keep, self._motion.shape[1])
        self._motion = self._motion[:, :keep]
        fkeep = min(max(self.frame_idx + 1 - self._frames_base, 0), len(self._frames))
        del self._frames[fkeep:]
        if cut_history and self._motion.shape[1] > self._patch:
            cut = self._motion.shape[1] - self._patch
            self._motion = self._motion[:, cut:]
            self._motion_base += cut

    def set_prompt(self, text, once=False):
        text = self.normalize_prompt(text)
        with self.lock:
            self._once_active = False
            if text == self.idle_prompt and self._motion is not None:
                # stop request: play the get-up transition first
                self.prompt = self.recover_prompt
                self._recover_at = self.frame_idx + int(self._recover_s * self.fps)
                self._recover_cap = self.frame_idx + int(self._recover_max_s * self.fps)
            else:
                self.prompt = text
                self._recover_at = None
                if once:
                    self._once_active = True
                    self._once_left = False
                    self._once_run = 0
                    self._once_deadline = self.frame_idx + int(self._once_max_s * self.fps)
            self._text_feat = self._encode(self.prompt)
            self._truncate_to_served(cut_history=True)
        return self.prompt

    def reset(self):
        """forget the rollout, next motion spawns fresh at the origin."""
        with self.lock:
            self._motion = None
            self._motion_base = 0
            self._frames = []
            self._frames_base = 0
            self.frame_idx = 0
            self.prompt = self.idle_prompt
            self._recover_at = None
            self._once_active = False
            self._text_feat = self._encode(self.idle_prompt)
            self._epoch += 1
            self._cond.notify_all()

    # -- generation (worker thread) --

    def _worker_loop(self):
        try:
            while True:
                with self._cond:
                    while (len(self._frames) - (self.frame_idx - self._frames_base)
                           >= self._lookahead):
                        self._cond.wait(0.05)
                    snap = self._snapshot()
                samples, frames = self._run_step(snap)
                with self._cond:
                    if snap['epoch'] == self._epoch:
                        self._append(samples, frames)
                        self._cond.notify_all()
                    # else: prompt/reset landed mid generation, discard
        except Exception as e:
            with self._cond:
                self._worker_err = e
                self._cond.notify_all()

    def _snapshot(self):
        """grab everything a generation step needs, under the lock. tensors
        are immutable once created (truncation reslices, append cats), so
        holding references is safe."""
        if self._motion is None:
            hist, hist_len = None, 0
            init_t = torch.zeros(1, 3, device=self.device)
            init_h = torch.zeros(1, device=self.device)
        else:
            hist_len = min(self._motion.shape[1], self._hist_cap) // self._patch * self._patch
            hist = self._motion[:, -hist_len:]
            init_t = init_h = None
        return {'epoch': self._epoch, 'text_feat': self._text_feat,
                'hist': hist, 'hist_len': hist_len, 'init_t': init_t, 'init_h': init_h}

    @torch.no_grad()
    def _run_step(self, snap):
        """the gpu heavy part, runs outside the lock."""
        hist_len = snap['hist_len']
        text_feat = snap['text_feat']
        mask = torch.ones(1, text_feat.shape[1], device=self.device, dtype=torch.bool)
        samples = self.model.autoregressive_step(
            num_frames=hist_len + self._horizon,
            num_denoising_steps=self._steps,
            motion_mask=None, observed_motion=None,
            cfg_weight=self._cfg,
            texts=None, text_feat=text_feat, text_pad_mask=mask,
            init_history_sequence=snap['hist'],
            init_global_translation=snap['init_t'],
            init_first_heading_angle=snap['init_h'],
        )
        out = self.model.motion_rep.inverse(
            self.model.motion_rep.unnormalize(samples), is_normalized=False)

        def np_of(key):
            return out[key][0, hist_len:].float().cpu().numpy()

        joints = np_of('posed_joints')
        rots = np_of('global_rot_mats')
        heading = np_of('global_root_heading')
        root = np_of('root_positions')
        smooth = np_of('smooth_root_pos')
        frames = [{'joints': joints[i], 'rotmats': rots[i], 'heading': heading[i],
                   'root_pos': root[i], 'smooth_root': smooth[i]}
                  for i in range(joints.shape[0])]
        return samples[:, hist_len:], frames

    def _append(self, new, frames):
        """attach a finished chunk, under the lock."""
        if self._motion is None:
            self._motion = new
        else:
            self._motion = torch.cat([self._motion, new], dim=1)
        self._frames.extend(frames)

        # trim tails so hour-long sessions dont grow without bound
        max_keep = max(self._hist_cap * 2, self._lookahead + self._hist_cap)
        if self._motion.shape[1] > max_keep:
            cut = self._motion.shape[1] - max_keep
            self._motion = self._motion[:, cut:]
            self._motion_base += cut
        served = self.frame_idx - self._frames_base
        if served > max_keep:
            cut = served - self._patch
            del self._frames[:cut]
            self._frames_base += cut

    def _upright(self):
        """is the currently served pose standing? hips near stand height and
        the pelvis up axis pointing up."""
        i = self.frame_idx - self._frames_base - 1
        if not (0 <= i < len(self._frames)):
            return True
        f = self._frames[i]
        return float(f['root_pos'][1]) > 0.75 and float(f['rotmats'][0][1, 1]) > 0.7

    def next_frame(self):
        """return the next frame as plain python data, waiting on the worker
        if the buffer is empty (fresh prompt, cold start)."""
        with self._cond:
            if self._recover_at is not None and self.frame_idx >= self._recover_at:
                if self._upright() or self.frame_idx >= self._recover_cap:
                    # get-up transition played out, settle into the real idle.
                    # standing history is fine to keep, so no history cut.
                    self._recover_at = None
                    self.prompt = self.idle_prompt
                    self._text_feat = self._encode(self.idle_prompt)
                    self._truncate_to_served(cut_history=False)
                else:
                    # still mid rise (lying takes longer), check again shortly
                    self._recover_at = self.frame_idx + self._patch
            if self._once_active:
                # one-shot action: done once he leaves the standing pose and
                # comes back to it for a settle period (a double flip stays
                # non-upright between the flips, so it completes both first)
                if not self._upright():
                    self._once_left = True
                    self._once_run = 0
                elif self._once_left:
                    self._once_run += 1
                    if self._once_run >= int(self._once_settle_s * self.fps):
                        self._once_active = False
                        self.prompt = self.idle_prompt
                        self._text_feat = self._encode(self.idle_prompt)
                        self._truncate_to_served(cut_history=False)
                if self._once_active and self.frame_idx >= self._once_deadline:
                    # never came back on its own (held a pose, or the action
                    # never leaves upright like waving), force the stop path
                    self._once_active = False
                    self.prompt = self.recover_prompt
                    self._recover_at = self.frame_idx + int(self._recover_s * self.fps)
                    self._recover_cap = self.frame_idx + int(self._recover_max_s * self.fps)
                    self._text_feat = self._encode(self.prompt)
                    self._truncate_to_served(cut_history=True)
            while self.frame_idx - self._frames_base >= len(self._frames):
                if self._worker_err is not None:
                    raise RuntimeError('ardy generation worker died') from self._worker_err
                self._cond.wait(0.1)
            f = self._frames[self.frame_idx - self._frames_base]
            idx = self.frame_idx
            self.frame_idx += 1
            self._cond.notify_all()  # buffer shrank, wake the worker
        return {'t': idx, **f}
