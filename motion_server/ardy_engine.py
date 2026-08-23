# ARDY streaming engine: same duck-typed surface as dart_engine.MotionEngine
# (fps / idle_prompt / normalize_prompt / set_prompt / reset / next_frame)
# but backed by nvidia ARDY core checkpoints instead of DART.
#
# streaming pattern mirrors the loop body of ardy's own long generation: the
# hybrid latent history is the state and persists across steps, each step
# denoises one horizon onto it and requantizes, and the explicit motion we
# stream out is a decoded copy that never feeds back in. text encoder is the
# community NF4 quant of the gated LLM2Vec llama so no meta approval is needed
# (~5.5GB vram, encodes a prompt in ~0.3s, cached).

import threading
from pathlib import Path

import numpy as np
import torch

ENCODER_PATH = str(Path(__file__).parent / 'text_encoders' / 'llm2vec_nf4')


class ArdyEngine:
    """owns the ARDY rollout state. all methods must be called from one thread."""

    def __init__(self, model='core8', device='cuda', steps=8, hist_cap_s=7.2,
                 cfg_text=2.0, cfg_constraint=2.0, postprocess=True,
                 contact_threshold=0.5, root_margin=0.04):
        # hist_cap_s is the STEADY STATE context fed back each step, and it
        # is a real tradeoff, measured in benchmarks/ardy_history.py over 3
        # runs each at 0.2 / 0.6 / 2.0s:
        #   long history smothers the text condition. a 'walks forward' loop
        #   averaged 0.87 m/s at 2.0s with its worst 5s window down at 0.43,
        #   ie he just stops walking mid prompt. at 0.6s thats 1.04 mean and
        #   0.90 worst, the loop actually keeps looping.
        #   short history loses the pose instead. holding a sit for 30s at
        #   0.2s had him standing back up 45% of the time, at 0.6s 0.7%.
        # 0.6s (3 tokens) is the only setting that wins both. the upstream
        # interactive demo defaults to 1 token, but it is driven by a human
        # retyping prompts, not by held poses.
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
        self._postprocess = bool(postprocess)
        self._contact_thresh = float(contact_threshold)
        self._root_margin = float(root_margin)

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
        # queued follow-up steps. one sentence describing several actions does
        # NOT work on core8: chained prompts freeze instead of sequencing, and
        # the freeze gets worse the more history you give it (measured 2s of
        # dead time at 0.6s history vs 19s at 7s). so a chain is played as
        # separate prompts, advancing when each one lands.
        self._queue = []
        self._step_until = None
        self._loop_last = False
        self._emb_cache = {}

        # a held prompt used to drift badly: |z| 6.3 -> 34.5 over three minutes
        # of sitting, bones stretched half a metre, ie the decoded joints
        # stopped describing a real body. the cause was feeding DECODED motion
        # back in as the next history, which re-runs the autoencoder both ways
        # every 0.4s. ardy's own long generation never does that, it keeps the
        # rollout in the hybrid latent and only decodes at the very end, so the
        # tokenizer round trip happens once instead of 150 times a minute.
        # we hold the same latent here and decode a throwaway copy to serve.
        self._hyb = None           # [1, T_tok, D] hybrid latent, the real state
        self._transl = None        # [1, 3] world offset the latent is centred against
        self._prompt_at = 0
        self._last_prompt = None

        # a full window of the OLD action outvotes a new prompt until enough of
        # it has scrolled out, which reads as him ignoring you for a few seconds.
        # so ramp: start a new action on a short context the text can steer, and
        # grow back to the cap once he is committed to it.
        self._resp_ramp = int(2.5 * self.fps)
        self._resp_floor = max(self._patch, int(0.8 * self.fps) // self._patch * self._patch)

        self._motion = None        # [1, T, D] normalized, tail of what was served
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
        # frames of already-generated motion kept past the playhead on a
        # prompt switch, to cover the ~300ms it takes to make the first chunk
        # of the new prompt. two tokens is 400ms at 20fps.
        self._replan_bridge = 2 * self._patch
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
        an attractor (long history dominates the text condition).

        keeps _replan_bridge frames past the playhead on purpose: cutting
        exactly at the playhead leaves nothing to serve while the first chunk
        of the new prompt generates, so every switch stalls ~300ms and then
        bursts. the bridge is old-prompt motion but it is only a fifth of a
        second and the new chunk conditions on it, so it joins cleanly."""
        self._epoch += 1  # void any chunk the worker has in flight
        self._cond.notify_all()
        if self._motion is None:
            return
        keep = max(self.frame_idx + 1 - self._motion_base + self._replan_bridge, self._patch)
        keep = min((keep + self._patch - 1) // self._patch * self._patch, self._motion.shape[1])
        self._motion = self._motion[:, :keep]
        # the latent has to lose the same frames, otherwise he carries on from a
        # future that never got served and the switch shows as a jump
        if self._hyb is not None:
            self._hyb = self._hyb[:, :max(1, keep // self._patch)]
        fkeep = min(max(self.frame_idx + 1 - self._frames_base + self._replan_bridge, 0),
                    len(self._frames))
        del self._frames[fkeep:]
        if cut_history and self._motion.shape[1] > self._patch:
            cut = self._motion.shape[1] - self._patch
            self._motion = self._motion[:, cut:]
            self._motion_base += cut
            if self._hyb is not None:
                self._hyb = self._hyb[:, -1:]

    def _begin_stop(self):
        """head for idle, returns whether history needs cutting.

        from a seated or lying pose 'stands still' describes a state the body
        is already in, so it just keeps sitting: play a get-up prompt first
        and cut history to break the attractor. already standing, go straight
        to idle, otherwise every stop costs a phantom crouch and rise, which
        the expression layer triggers constantly between gestures."""
        if self._upright():
            self.prompt = self.idle_prompt
            self._recover_at = self._recover_cap = None
            self._text_feat = self._encode(self.prompt)
            return False
        self.prompt = self.recover_prompt
        self._recover_at = self.frame_idx + int(self._recover_s * self.fps)
        self._recover_cap = self.frame_idx + int(self._recover_max_s * self.fps)
        self._text_feat = self._encode(self.prompt)
        return True

    def _apply_step(self, prompt, once, seconds):
        self.prompt = prompt
        self._recover_at = None
        self._once_active = bool(once)
        self._once_left = False
        self._once_run = 0
        self._once_deadline = self.frame_idx + int(self._once_max_s * self.fps)
        self._step_until = (self.frame_idx + int(seconds * self.fps)
                            if seconds else None)
        self._text_feat = self._encode(prompt)

    def _advance(self):
        """move to the next queued step, returns False if the queue is empty."""
        if not self._queue:
            return False
        prompt, once, seconds = self._queue.pop(0)
        hold = not self._queue and self._loop_last
        self._apply_step(prompt, once and not hold, None if hold else seconds)
        self._truncate_to_served(cut_history=True)
        return True

    def set_sequence(self, steps, seconds_each=None, loop_last=False):
        """play prompts back to back, each advancing when it lands or times out."""
        parsed = []
        for s in steps:
            if isinstance(s, str):
                prompt, secs = s, seconds_each
            else:
                prompt = s.get('prompt', '')
                secs = s.get('seconds', seconds_each)
            prompt = self.normalize_prompt(prompt)
            parsed.append((prompt, True, float(secs) if secs else None))
        if not parsed:
            return []
        with self.lock:
            self._loop_last = bool(loop_last)
            self._queue = parsed[1:]
            prompt, once, secs = parsed[0]
            hold = not self._queue and self._loop_last
            self._apply_step(prompt, once and not hold, None if hold else secs)
            self._truncate_to_served(cut_history=True)
        return [p for p, _, _ in parsed]

    def set_prompt(self, text, once=False):
        text = self.normalize_prompt(text)
        with self.lock:
            self._once_active = False
            self._queue = []
            self._step_until = None
            if text == self.idle_prompt and self._motion is not None:
                cut = self._begin_stop()
            else:
                cut = True
                self.prompt = text
                self._recover_at = None
                if once:
                    self._once_active = True
                    self._once_left = False
                    self._once_run = 0
                    self._once_deadline = self.frame_idx + int(self._once_max_s * self.fps)
                self._text_feat = self._encode(self.prompt)
            self._truncate_to_served(cut_history=cut)
        return self.prompt

    def set_tuning(self, history=None, steps=None, postprocess=None,
                   contact_threshold=None, root_margin=None, reanchor=None,
                   cfg_text=None, ramp=None):
        """live knobs so this can be dialled in from the client while in game."""
        with self.lock:
            if history is not None:
                self._hist_cap = max(self._patch,
                                     int(float(history) * self.fps) // self._patch * self._patch)
            if cfg_text is not None:
                self._cfg = (float(cfg_text), self._cfg[1])
            if ramp is not None:
                self._resp_ramp = max(0, int(float(ramp) * self.fps))
            if steps is not None:
                base = int(getattr(getattr(self.model, 'diffusion', None), 'num_base_steps', 10) or 10)
                self._steps = max(1, min(int(steps), base))
            if postprocess is not None:
                self._postprocess = bool(postprocess)
            if contact_threshold is not None:
                self._contact_thresh = float(contact_threshold)
            if root_margin is not None:
                self._root_margin = float(root_margin)
            if reanchor is not None:
                pass  # kept so older clients dont error, the latent requantize replaced it
            self._epoch += 1  # void the in-flight chunk so changes land now
            self._cond.notify_all()
        return self.tuning()

    def tuning(self):
        return {'history_s': round(self._hist_cap / self.fps, 2),
                'history_frames': self._hist_cap,
                'steps': self._steps,
                'postprocess': self._postprocess,
                'contact_threshold': self._contact_thresh,
                'root_margin': self._root_margin,
                'cfg_text': self._cfg[0],
                'ramp_s': round(self._resp_ramp / self.fps, 2)}

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
            self._queue = []
            self._step_until = None
            self._hyb = None
            self._transl = None
            self._last_prompt = None   # re-bases _prompt_at off frame 0
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
                samples, frames, hyb, transl = self._run_step(snap)
                with self._cond:
                    if snap['epoch'] == self._epoch:
                        self._append(samples, frames, hyb, transl)
                        self._cond.notify_all()
                    # else: prompt/reset landed mid generation, discard
        except Exception as e:
            with self._cond:
                self._worker_err = e
                self._cond.notify_all()

    def _hist_cap_now(self):
        """how much context this step gets, see _resp_ramp in __init__."""
        age = self.frame_idx - self._prompt_at
        if not self._resp_ramp or age >= self._resp_ramp:
            return self._hist_cap
        lo = min(self._resp_floor, self._hist_cap)
        return lo + (self._hist_cap - lo) * age // self._resp_ramp

    def _snapshot(self):
        """grab everything a generation step needs, under the lock. tensors
        are immutable once created (truncation reslices, append cats), so
        holding references is safe."""
        if self.prompt != self._last_prompt:
            self._last_prompt = self.prompt
            self._prompt_at = self.frame_idx
        want_tok = max(1, self._hist_cap_now() // self._patch)
        return {'epoch': self._epoch, 'text_feat': self._text_feat,
                'hyb': self._hyb, 'transl': self._transl, 'want_tok': want_tok}

    @torch.no_grad()
    def _run_step(self, snap):
        """one autoregressive window, mirroring the loop body of ardy's own
        __call__ rather than their single shot autoregressive_step helper.

        the difference is what carries between steps. autoregressive_step takes
        EXPLICIT motion in and hands EXPLICIT motion back, so driving it in a
        loop tokenizes and detokenizes every 0.4s; encode(decode(z)) != z and
        the error feeds straight into the next step. their long generation
        keeps the hybrid latent across every step and decodes once at the end.
        we do the same, and treat the decode purely as output."""
        from ardy.model.ardy_model import translate_normalized_root_motion
        model = self.model
        rep = model.motion_rep
        hybrid = model.hybrid
        patch, horizon = self._patch, self._horizon
        text_feat = snap['text_feat']
        mask = torch.ones(1, text_feat.shape[1], device=self.device, dtype=torch.bool)

        hyb, transl = snap['hyb'], snap['transl']
        if transl is None:
            transl = torch.zeros(1, rep.nfeats_dict['root_pos'], device=self.device)
        if hyb is None:
            hist_frames = 0
            heading = torch.zeros(1, device=self.device)
        else:
            hyb = hyb[:, -snap['want_tok']:]
            hist_frames = hyb.shape[1] * patch
            # the denoiser wants the heading of the first frame of the window it
            # is actually given, so read it back off the window we sliced
            root, _ = hybrid.get_root_and_latent_body_motion_from_hybrid(hyb)
            heading = rep.get_root_heading_angle(
                rep.global_root_stats.unnormalize(root))[:, 0]

        steps_t = torch.tensor([self._steps], device=self.device)
        use_timesteps = model.diffusion.space_timesteps(steps_t[0])[0]
        model.diffusion.calc_diffusion_vars(use_timesteps)
        indices = list(range(self._steps))[::-1]

        hyb = model._generate_window(
            hyb, transl, 0, hist_frames, hist_frames + horizon,
            text_feat, mask, heading, None, None,
            steps_t, self._cfg, indices,
            progress_bar=lambda it: it, target_motion=None, cfg_type=None,
        )
        # recentring on the newest frame keeps the root features small, and the
        # requantize is the bit that matters: it snaps the body latents back
        # onto the codebook grid every step, so they cannot slowly wander off
        # the manifold the decoder was trained on.
        center = torch.full((1,), hist_frames + horizon - 1,
                            device=self.device, dtype=torch.long)
        hyb, center_pos, _ = model._recenter_history(hyb, center, requantize=True)
        transl = transl + center_pos

        # decode a throwaway copy in world space, the latent above is the state
        root, body = hybrid.get_root_and_latent_body_motion_from_hybrid(hyb)
        world = hybrid.get_hybrid_motion_from_root_and_latent_body_motion(
            translate_normalized_root_motion(root, transl, rep), body)
        nframes = hyb.shape[1] * patch
        samples = hybrid.get_explicit_motion_from_hybrid(
            world,
            torch.ones(1, nframes, device=self.device, dtype=torch.bool),
            torch.full((1,), nframes, device=self.device, dtype=torch.long),
            motion_mask=None,
        )
        hist_len = nframes - horizon
        out = rep.inverse(rep.unnormalize(samples), is_normalized=False)

        if self._postprocess:
            # ardy's own foot-contact cleanup, on by default in their generate.py
            # for every model but g1. it plants the feet (they otherwise sink and
            # skate) and pulls the root back to the ground plane. output only,
            # deliberately: correcting explicit motion cannot be folded back into
            # the latent without re-encoding, which is the thing that broke it.
            from ardy.postprocess import post_process_motion
            fix = post_process_motion(
                out['local_rot_mats'][:, hist_len:],
                out['root_positions'][:, hist_len:],
                out['foot_contacts'][:, hist_len:],
                rep.skeleton,
                contact_threshold=self._contact_thresh,
                root_margin=self._root_margin,
            )
            fixed = rep.normalize(rep(local_joint_rots=fix['local_rot_mats'],
                                      root_positions=fix['root_positions'],
                                      to_normalize=False))
            samples = torch.cat([samples[:, :hist_len], fixed], dim=1)
            # re-decode so heading and contacts agree with the corrected pose
            out = rep.inverse(rep.unnormalize(samples), is_normalized=False)

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
        return samples[:, hist_len:], frames, hyb, transl

    def _append(self, new, frames, hyb, transl):
        """attach a finished chunk, under the lock."""
        self._hyb = hyb
        self._transl = transl
        if self._motion is None:
            self._motion = new
        else:
            self._motion = torch.cat([self._motion, new], dim=1)
        self._frames.extend(frames)

        # trim tails so hour-long sessions dont grow without bound. cuts stay
        # patch aligned so _motion and _hyb keep indexing the same frames.
        max_keep = max(self._hist_cap * 2, self._lookahead + self._hist_cap)
        max_keep = max(max_keep // self._patch * self._patch, self._patch)
        if self._motion.shape[1] > max_keep:
            cut = (self._motion.shape[1] - max_keep) // self._patch * self._patch
            if cut:
                self._motion = self._motion[:, cut:]
                self._motion_base += cut
        keep_tok = max_keep // self._patch
        if self._hyb.shape[1] > keep_tok:
            self._hyb = self._hyb[:, -keep_tok:]
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
            if self._step_until is not None and self.frame_idx >= self._step_until:
                self._step_until = None
                if not self._advance():
                    self._once_active = False
                    self._truncate_to_served(cut_history=self._begin_stop())
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
                        if not self._advance():
                            self.prompt = self.idle_prompt
                            self._text_feat = self._encode(self.idle_prompt)
                            self._truncate_to_served(cut_history=False)
                if self._once_active and self.frame_idx >= self._once_deadline:
                    # never came back on its own (held a pose, or the action
                    # never leaves upright like waving), force the stop path
                    self._once_active = False
                    if not self._advance():
                        self._truncate_to_served(cut_history=self._begin_stop())
            while self.frame_idx - self._frames_base >= len(self._frames):
                if self._worker_err is not None:
                    raise RuntimeError('ardy generation worker died') from self._worker_err
                self._cond.wait(0.1)
            f = self._frames[self.frame_idx - self._frames_base]
            idx = self.frame_idx
            self.frame_idx += 1
            self._cond.notify_all()  # buffer shrank, wake the worker
        return {'t': idx, **f}
