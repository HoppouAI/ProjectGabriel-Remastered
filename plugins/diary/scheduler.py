"""Background scheduler that ticks the diary summarizer every N seconds.

Single asyncio task that lives for the lifetime of the host process, kicked
off from the plugin's `startup` event subscriber. Cancelled on shutdown.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from .diary import DiaryStore
from .summarizer import DEFAULT_MODEL, write_next_entry

logger = logging.getLogger(__name__)


class DiaryScheduler:
    def __init__(
        self,
        store: DiaryStore,
        conv_dir: Path,
        get_api_key: Callable[[], str],
        interval_seconds: float,
        max_sessions: int = 5,
        model: str = DEFAULT_MODEL,
        initial_delay_seconds: float = 300.0,
        get_persona: Optional[Callable[[], str]] = None,
    ):
        self.store = store
        self.conv_dir = conv_dir
        self.get_api_key = get_api_key
        self.interval = max(60.0, float(interval_seconds))
        self.max_sessions = max_sessions
        self.model = model
        self.initial_delay = max(0.0, float(initial_delay_seconds))
        self.get_persona = get_persona or (lambda: "")
        self._task: Optional[asyncio.Task] = None
        self._stop = False

    def start(self) -> bool:
        """Schedule the loop on the running event loop. Safe to call from a
        sync context as long as a loop is running (the plugin startup event
        fires inside the main asyncio loop)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("diary scheduler: no running loop, not starting")
            return False
        if self._task is not None and not self._task.done():
            return True
        self._stop = False
        self._task = loop.create_task(self._run())
        logger.info(f"diary scheduler: every {self.interval / 60:.0f} min, after initial {self.initial_delay:.0f}s warmup")
        return True

    def stop(self) -> None:
        self._stop = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def tick_once(self) -> bool:
        """Run a single summarization pass. Returns True if an entry was written."""
        api_key = ""
        try:
            api_key = self.get_api_key() or ""
        except Exception as e:
            logger.warning(f"diary: get_api_key raised: {e}")
            return False
        if not api_key:
            logger.debug("diary: no api key available, skipping tick")
            return False
        try:
            persona = ""
            try:
                persona = self.get_persona() or ""
            except Exception as e:
                logger.warning(f"diary: get_persona raised: {e}")
            entry = await write_next_entry(
                api_key=api_key,
                store=self.store,
                conv_dir=self.conv_dir,
                max_sessions=self.max_sessions,
                model=self.model,
                persona=persona,
            )
        except Exception as e:
            logger.error(f"diary: tick failed: {e}")
            return False
        return entry is not None

    async def _run(self):
        try:
            if self.initial_delay > 0:
                await asyncio.sleep(self.initial_delay)
            while not self._stop:
                wrote = await self.tick_once()
                if wrote:
                    logger.info("diary: scheduler wrote a new entry")
                # sleep the full interval even on failures, dont hammer the API
                try:
                    await asyncio.sleep(self.interval)
                except asyncio.CancelledError:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"diary scheduler crashed: {e}")
        finally:
            logger.debug("diary scheduler stopped")
