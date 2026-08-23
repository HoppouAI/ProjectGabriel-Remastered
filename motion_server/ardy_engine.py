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

# words that count as the caption already having a subject. bare 'person' in
# the string is not enough: 'punches a man' still needs one, while 'a tall man
# runs' already has one, so the noun only counts at the head of the sentence.
_SUBJECT_NOUNS = frozenset((
    'person', 'man', 'woman', 'guy', 'girl', 'human', 'boy', 'lady',
    'male', 'female', 'figure', 'character', 'dude',
))
_SUBJECT_PRONOUNS = frozenset(('he', 'she', 'they', 'someone', 'somebody'))
_DETERMINERS = frozenset(('a', 'an', 'the', 'this', 'that', 'one'))


def has_subject(text):
    w = text.lower().split()
    if not w:
        return False
    if w[0] in _SUBJECT_PRONOUNS:
        return True
    if w[0] in _DETERMINERS:
        return any(x in _SUBJECT_NOUNS for x in w[1:3])
    return False


class ArdyEngine:
    """owns the ARDY rollout state. all methods must be called from one thread."""

    def __init__(self, model='core8', device='cuda', steps=8, hist_cap_s=0.6,
                 cfg_text=2.0, cfg_constraint=2.0):
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

        # a held prompt drifts. the rollout only ever sees its own output
        # through a 12 frame window, so small biases compound with nothing to
        # correct them: holding a sit for 3 minutes took the normalized latent
        # from |z| 6.3 to 34.5 and stretched bone lengths by half a metre, ie
        # the decoded joints stopped describing a real body. static poses are
        # worst because a near constant history carries almost no signal, a
        # gait cycle at least keeps re-injecting structure.
        # there is no absolute limit to clamp against, healthy |z| is prompt
        # dependent (stand 1.5, sit 6.5, run 18.6), so instead keep a known
        # good conditioning window from just after the prompt landed and fall
        # back to it on a leash.
        self._anchor = None
        self._anchor_prompt = None
        self._anchor_z = 0.0
        self._anchor_at = 0
        self._prompt_at = 0
        self._anchor_after = int(4.0 * self.fps)
        self._anchor_settle = int(8.0 * self.fps)
        self._reanchor_every = int(15.0 * self.fps)

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
        """ardy wants humanml style captions, same as the hml3d dart variant.

        only gives the caption a subject if it hasn't got one. the old test was
        'person' not in text, so 'a man runs' went in as 'a person a man runs',
        which is why asking for a gendered gait never appeared to do anything."""
        text = ' '.join(str(text).replace('.', ' ').replace('!', ' ').split())
        if not text:
            return self.idle_prompt
        if not has_subject(text):
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
        keep = min(keep, self._motion.shape[1])
        self._motion = self._motion[:, :keep]
        fkeep = min(max(self.frame_idx + 1 - self._frames_base + self._replan_bridge, 0),
                    len(self._frames))
        del self._frames[fkeep:]
        if cut_history and self._motion.shape[1] > self._patch:
            cut = self._motion.shape[1] - self._patch
            self._motion = self._motion[:, cut:]
            self._motion_base += cut

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
            self._anchor = None
            self._anchor_prompt = None  # also re-bases _prompt_at off frame 0
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

    def _rebase(self, anchor):
        """put an anchor window back at the current world pose.

        the history tensor carries absolute planar position and heading, and
        autoregressive_step reads them straight off it (_encode_init_history
        recenters on the last history frame and ignores init_global_translation
        entirely when a history is passed). so replaying an older window as-is
        rewinds him through the world: measured a 14m jump in a single frame
        mid walk. rotate then translate so the swap is invisible."""
        rep = self.model.motion_rep
        a = rep.unnormalize(anchor)
        cur = rep.unnormalize(self._motion[:, -1:])
        a = rep.rotate(a, rep.get_root_heading_angle(cur)[:, 0]
                       - rep.get_root_heading_angle(a)[:, -1])
        delta = (rep.get_root_pos(cur)[:, 0][:, [0, 2]]
                 - rep.get_root_pos(a)[:, -1][:, [0, 2]])
        return rep.normalize(rep.translate_2d(a, delta))

    def _guard_history(self, hist):
        """keep the conditioning window inside the distribution the model was
        trained on, see the _anchor note in __init__.

        the anchor is captured once the prompt has landed, then reused either
        on a timer or as soon as |z| runs away from where it started. the
        threshold is relative because a healthy run sits higher than a broken
        sit does."""
        if self.prompt != self._anchor_prompt:
            self._anchor = None
            self._anchor_prompt = self.prompt
            self._prompt_at = self.frame_idx
        z = float(hist.abs().max())
        age = self.frame_idx - self._prompt_at
        if age < self._anchor_after:
            # prompt still landing, nothing here is representative yet
            self._anchor = None
            self._anchor_z = 0.0
            return hist
        if age < self._anchor_settle:
            # learn what healthy looks like for THIS prompt over a window, not
            # at one instant: 4s into a walk he is still accelerating, and a
            # baseline taken there reads 5.8 against a steady state of 10-12,
            # which trips the divergence test on a perfectly good gait.
            self._anchor = hist.clone()
            self._anchor_z = max(self._anchor_z, z)
            self._anchor_at = self.frame_idx
            return hist
        if self._anchor is None:
            return hist
        if (z > max(2.0 * self._anchor_z, self._anchor_z + 5.0)
                or self.frame_idx - self._anchor_at >= self._reanchor_every):
            # _motion is conditioning only, never served, so rewriting it is
            # invisible downstream. the base keeps _truncate_to_served honest.
            self._motion = self._rebase(self._anchor)
            self._motion_base = self.frame_idx + 1 - self._motion.shape[1]
            self._anchor_at = self.frame_idx
            return self._motion
        return hist

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
            hist = self._guard_history(self._motion[:, -hist_len:])
            hist_len = hist.shape[1]
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
