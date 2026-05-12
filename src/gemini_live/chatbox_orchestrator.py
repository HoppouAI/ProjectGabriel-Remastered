"""Chatbox orchestrator.

Centralizes the rules for who gets to write to the VRChat chatbox at any
given moment. Built-in displays (local music, Lyria music gen) always
beat plugin contributed displays. Within each tier, lower priority wins.

Why this exists: the previous flat loop in session.py had a few bugs that
plugin authors kept hitting:

  1. When a plugin source flipped is_active() False the previous text
     would stay visible until something else wrote, so an empty 'now
     playing' line could linger after a song ended.
  2. is_active() / render() exceptions disabled iteration only at debug
     level for one tick, no warning surfaced and no automatic disable
     for genuinely broken sources.
  3. Same text rewritten every 1.3s regardless of whether anything
     changed.
  4. No coordination between idle banner and active source -- the idle
     banner could re-enter a frame after the active source had already
     written, blanking the active text for one tick.

The orchestrator runs the per-tick decision in one place. Both the
session's now_playing_loop and the idle_check_loop go through it so
they cant fight each other.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# After this many consecutive exceptions a source is suspended for the
# rest of the run. Surfaces as a single warning to the plugin issue log,
# does not crash the app.
_MAX_SOURCE_FAILURES = 5

# How long to keep the same text on screen before forcing a refresh.
# VRChat's chatbox times out and clears itself after about 9 seconds of
# silence, so we re-send slightly under that as a keepalive.
_FORCE_REFRESH_SECONDS = 6.0


class ChatboxSource:
    """Optional helper base. Plugin sources can subclass this for the
    common pattern, but anything quack-typed with is_active/render works
    too. The orchestrator never isinstance-checks this."""

    def is_active(self) -> bool:
        return False

    def render(self) -> Optional[str]:
        return None

    def on_clear(self) -> None:
        """Called once when this source loses the chatbox to a different
        winner OR transitions to inactive with nothing to take over.
        Safe to override, default is a no-op."""
        pass


class ChatboxOrchestrator:
    """One-shot decider. Built once per session and ticked from the
    now-playing loop.

    Flow per tick:

      1. Walk built-in displays first (local music, music_gen).
      2. If no built-in active, walk plugin sources in priority order.
      3. If a winner emerged, render and send (deduped).
      4. If the previous winner is gone and nobody else stepped in,
         emit an on_clear() to the previous winner so it can do
         cleanup, mark the chatbox as released, and let the idle banner
         take over.

    Built-in display callables return either:
      - a (name: str, text: str) tuple -- name must be unique per builtin
      - None  -- this builtin is not active right now
    """

    def __init__(self, send_chatbox: Callable[[str], Any]):
        self._send = send_chatbox
        self._builtins: list[tuple[str, Callable[[], Optional[tuple[str, str]]]]] = []
        # Per-source consecutive failure counts and a denylist of
        # sources we've given up on.
        self._failures: dict[str, int] = {}
        self._suspended: set[str] = set()
        # Bookkeeping for what we last sent so we can dedupe and detect
        # a winner transition.
        self._last_winner: Optional[str] = None
        self._last_text: Optional[str] = None
        self._last_send: float = 0.0
        # Reference back to the active source object so we can call
        # on_clear when the winner changes. Stored separately from the
        # name because plugin sources might be re-registered with the
        # same name pointing at a fresh object.
        self._last_source_obj: Any = None

    def register_builtin(
        self,
        name: str,
        producer: Callable[[], Optional[tuple[str, str]]],
    ) -> None:
        """Register a host-internal display source. These are checked
        before plugin sources every tick. Producer should return either
        a (label, text) tuple when active or None when not."""
        self._builtins.append((name, producer))

    def has_active_source(self) -> bool:
        """True when any built-in or plugin source claims it wants the
        chatbox right now. Used by the idle loop to know whether to
        suppress the idle banner."""
        for _name, prod in self._builtins:
            try:
                res = prod()
                if res is not None:
                    return True
            except Exception as e:
                logger.debug(f"builtin chatbox '{_name}' is_active raised: {e}")
        return self._any_plugin_active()

    def tick(self) -> None:
        """Pick a winner and send. Called from now_playing_loop ~1Hz."""
        winner: Optional[tuple[str, str, Any]] = None  # (kind, text, source_obj)

        # builtins first
        for name, prod in self._builtins:
            try:
                res = prod()
            except Exception as e:
                logger.debug(f"builtin chatbox '{name}' producer raised: {e}")
                continue
            if res is None:
                continue
            try:
                label, text = res
            except (TypeError, ValueError):
                logger.debug(f"builtin chatbox '{name}' returned non-tuple: {res!r}")
                continue
            if not text:
                continue
            winner = (f"builtin:{label}", text, None)
            break

        # plugin sources, lowest priority first
        if winner is None:
            for name, src in self._iter_plugin_sources():
                if not self._call_active(name, src):
                    continue
                text = self._call_render(name, src)
                if not text:
                    continue
                winner = (f"plugin:{name}", text, src)
                break

        now = time.time()

        if winner is None:
            self._handle_no_winner()
            return

        wname, wtext, wobj = winner

        # If winner changed, give the previous source an on_clear callback
        if self._last_winner is not None and self._last_winner != wname:
            self._fire_on_clear(self._last_source_obj)

        # Dedupe: only send if text changed or it's been long enough
        # that we should re-send to keep the chatbox alive.
        text_changed = wtext != self._last_text
        force_refresh = (now - self._last_send) >= _FORCE_REFRESH_SECONDS
        if text_changed or force_refresh or self._last_winner != wname:
            try:
                self._send(wtext)
            except Exception as e:
                logger.error(f"chatbox send failed for {wname}: {e}")
                return
            self._last_text = wtext
            self._last_send = now

        self._last_winner = wname
        self._last_source_obj = wobj

    def _handle_no_winner(self) -> None:
        if self._last_winner is None:
            return
        # The previous winner is gone. Tell it via on_clear so it can
        # release any held UI state, but DO NOT blast a clear text to the
        # chatbox -- the idle loop will pick up next tick and write its
        # banner. Sending an empty string here can cause a one-frame
        # flicker between the active text disappearing and the banner
        # arriving.
        self._fire_on_clear(self._last_source_obj)
        self._last_winner = None
        self._last_text = None
        self._last_source_obj = None

    def _fire_on_clear(self, src: Any) -> None:
        if src is None:
            return
        on_clear = getattr(src, "on_clear", None)
        if not callable(on_clear):
            return
        try:
            on_clear()
        except Exception as e:
            logger.debug(f"chatbox source on_clear raised: {e}")

    def _iter_plugin_sources(self):
        from src.plugins.api import iter_chatbox_sources
        for name, src in iter_chatbox_sources():
            if name in self._suspended:
                continue
            yield name, src

    def _any_plugin_active(self) -> bool:
        for name, src in self._iter_plugin_sources():
            if self._call_active(name, src):
                return True
        return False

    def _call_active(self, name: str, src: Any) -> bool:
        try:
            return bool(src.is_active())
        except Exception as e:
            self._record_failure(name, "is_active", e)
            return False

    def _call_render(self, name: str, src: Any) -> Optional[str]:
        try:
            text = src.render()
        except Exception as e:
            self._record_failure(name, "render", e)
            return None
        if text is None:
            return None
        return str(text)

    def _record_failure(self, name: str, where: str, exc: Exception) -> None:
        n = self._failures.get(name, 0) + 1
        self._failures[name] = n
        if n >= _MAX_SOURCE_FAILURES and name not in self._suspended:
            self._suspended.add(name)
            logger.warning(
                f"chatbox source '{name}' suspended after {n} consecutive "
                f"failures in {where}: {exc}"
            )
            try:
                from src.plugins.loader import record_plugin_issue
                record_plugin_issue(
                    name,
                    "warning",
                    f"chatbox source suspended after {n} failures in {where}: {exc}",
                )
            except Exception:
                pass
        else:
            logger.debug(f"chatbox source '{name}' {where} raised: {exc}")
