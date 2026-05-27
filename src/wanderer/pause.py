"""Pause/resume hooks driven by speech activity."""

from __future__ import annotations

import threading


class PauseMixin:
    def pause(self):
        with self._lock:
            self._paused = True
            self._zero_osc()

    def resume(self):
        with self._lock:
            self._paused = False
            self._auto_paused = False
            self._cancel_resume_timer()

    def on_speech_activity(self):
        if not self._active or self._paused:
            return
        with self._lock:
            self._paused = True
            self._auto_paused = True
            self._zero_osc()
        self._reset_resume_timer()

    def on_ai_speaking(self):
        if self._active and self._paused and self._auto_paused:
            self._reset_resume_timer()

    def _reset_resume_timer(self):
        self._cancel_resume_timer()
        t = threading.Timer(self._resume_delay, self._auto_resume)
        t.daemon = True
        t.start()
        self._resume_timer = t

    def _cancel_resume_timer(self):
        t = self._resume_timer
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass
            self._resume_timer = None

    def _auto_resume(self):
        if self._active and self._paused and self._auto_paused:
            self.resume()
