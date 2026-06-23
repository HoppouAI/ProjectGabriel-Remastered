"""Yap mode -- bring back the OG uninterruptible monologue tool.

Old gabriel had this so he could finish a sentence without people
talking over him. This one is smarter: it mutes the mic the same way
the existing voice tool does, then auto unmutes itself when the model
actually stops speaking, instead of relying on a dumb 60s timer.

Works with both VAD modes:
  - automatic VAD: muting the mic just stops audio from reaching the
    server, so it cant decide we got interrupted.
  - silero (3rd party) VAD: muting drops chunks before silero ever
    sees them, so no activity_start fires while yap mode is on.

When the model finishes a turn yap mode auto disables. The model can
re-enable it any time it wants another uninterrupted rant.
"""
from __future__ import annotations

import asyncio
import logging
import time

from google.genai import types

from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# safety net so we never lock the mic forever if upstream state gets weird
HARD_CAP_SECONDS = 120.0
# how long to wait for the model to actually start speaking after the
# tool call returns. it almost always starts within a second but give it
# some slack on slow networks
SPEAK_START_TIMEOUT = 8.0
# debounce silence before we call it "done" (avoids ending yap mode in
# the middle of a sentence pause)
SILENCE_DEBOUNCE = 1.0


@register_tool
class YapModeTool(BaseTool):
    tool_key = "yap_mode"

    def __init__(self, handler):
        super().__init__(handler)
        self._enabled = False
        self._reason = ""
        self._watcher_task: asyncio.Task | None = None

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="enableYapMode",
                description=(
                    "Mute the microphone so nobody can interrupt you while you talk. "
                    "Yap mode auto-disables the moment you finish speaking, and you "
                    "can re-enable it again instantly any time you want another "
                    "uninterrupted moment. Pairs really well with setVoiceBoost for "
                    "loud yelling rants where you do NOT want anyone cutting in."
                    "\n**Invocation Condition:** Call this aggressively any time "
                    "someone is talking over you, interrupting, being annoying, or "
                    "trying to derail what you were saying. Also call it when you "
                    "are about to deliver a monologue, story, rant, song, dramatic "
                    "speech, or comeback you want to finish in full. When in doubt "
                    "and the energy is high, turn it on. Skip it only for chill "
                    "back-and-forth chat where interruptions are fine."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "reason": {
                            "type": "STRING",
                            "description": "Short reason for enabling yap mode, used in logs.",
                        },
                    },
                },
            ),
            types.FunctionDeclaration(
                name="getYapModeStatus",
                description=(
                    "Check whether yap mode is currently active."
                    "\n**Invocation Condition:** Call when you need to know if the "
                    "mic is currently gated by yap mode."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name == "enableYapMode":
            return await self._enable(args.get("reason", ""))
        if name == "getYapModeStatus":
            return {"result": "ok", "enabled": self._enabled, "reason": self._reason}
        return None

    async def _enable(self, reason: str):
        sess = self.live_session
        if sess is None:
            return {"result": "error", "message": "no live session yet"}
        self._reason = (reason or "uninterrupted speaking").strip()
        try:
            sess.set_mic_muted(True)
        except Exception as e:
            logger.error(f"yap mode: set_mic_muted(True) failed: {e}")
            return {"result": "error", "message": str(e)}
        if not self._enabled:
            self._enabled = True
            logger.info(f"Yap mode ENABLED: {self._reason}")
        # restart the watcher so each new enable gets a fresh debounce
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
        self._watcher_task = asyncio.create_task(self._watch_until_silent())
        return {
            "result": "ok",
            "message": f"yap mode on, reason: {self._reason}",
            "enabled": True,
        }

    async def _disable(self, why: str):
        sess = self.live_session
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
        self._watcher_task = None
        if sess is not None:
            try:
                sess.set_mic_muted(False)
            except Exception as e:
                logger.error(f"yap mode: set_mic_muted(False) failed: {e}")
            # silero-mode safety: if the manual VAD was mid-utterance when
            # we muted, the silero loop never saw a fresh chunk to close
            # the activity. nudge it closed so the server doesnt sit
            # waiting on an open activity.
            try:
                if (
                    getattr(sess.config, "vad_mode", None) == "silero"
                    and getattr(sess, "_manual_vad_speaking", False)
                    and getattr(sess, "_session", None) is not None
                ):
                    await sess._send_activity_end(sess._session)
                    await sess._send_audio_stream_end(sess._session)
            except Exception as e:
                logger.debug(f"yap mode: silero activity_end cleanup failed: {e}")
        was = self._enabled
        if was:
            self._enabled = False
            logger.info(f"Yap mode DISABLED ({why})")
        self._reason = ""
        return {"result": "ok", "message": "yap mode off", "enabled": False, "was_enabled": was}

    async def _watch_until_silent(self):
        """Wait for the model to start (or be) speaking, then for it to
        fall silent for SILENCE_DEBOUNCE seconds, then auto-disable."""
        sess = self.live_session
        if sess is None:
            return

        def _ai_talking() -> bool:
            # _speaking only tracks the llm, tts audio keeps going after it
            # stops so peek at playback + the queued pcm aswell
            if getattr(sess, "_speaking", False):
                return True
            if getattr(sess, "_audio_playing", False):
                return True
            q = getattr(sess, "_audio_in_queue", None)
            if q is not None:
                try:
                    if not q.empty():
                        return True
                except Exception:
                    pass
            return False

        try:
            # phase 1: wait for speaking to begin (usually nearly instant)
            t0 = time.monotonic()
            while not _ai_talking():
                if time.monotonic() - t0 > SPEAK_START_TIMEOUT:
                    # model never started talking, bail out gracefully
                    logger.info("yap mode: model never started speaking, auto-disabling")
                    await self._disable("never started speaking")
                    return
                await asyncio.sleep(0.1)

            # phase 2: wait for silence (debounced) or hard cap
            hard_cap_at = time.monotonic() + HARD_CAP_SECONDS
            silent_since = 0.0
            while True:
                if not _ai_talking():
                    now = time.monotonic()
                    if silent_since == 0.0:
                        silent_since = now
                    elif now - silent_since >= SILENCE_DEBOUNCE:
                        break
                else:
                    silent_since = 0.0
                if time.monotonic() > hard_cap_at:
                    logger.warning("yap mode: hit %.0fs hard cap, force-disabling", HARD_CAP_SECONDS)
                    break
                await asyncio.sleep(0.1)

            await self._disable("ai finished speaking")
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("yap mode watcher crashed, force-disabling")
            try:
                await self._disable("watcher error")
            except Exception:
                pass
