"""Background loop: reflects on finished sessions, decays old lessons, and reminds him of promises when people show up."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path

from .prompt import build_mind_block
from .reflect import Reflector
from .store import MindStore

logger = logging.getLogger(__name__)

_current: MindRuntime | None = None


def current() -> MindRuntime | None:
    return _current


class MindRuntime:
    def __init__(self, config, session=None, instance_monitor=None, store: MindStore | None = None,
                 memory=None, generate=None):
        global _current
        self.config = config
        self.session = session
        self.instance_monitor = instance_monitor
        self.store = store or MindStore(config.mind("db_path", "data/mind.sqlite"))
        if memory is None:
            from src.memory import memory_system
            memory = memory_system
        if generate is None:
            from .llm import generate_json
            model = config.mind("model", "gemini-3.5-flash-lite")

            async def generate(system, prompt):
                return await generate_json(config, model, system, prompt)
        self.half_life = float(config.mind("lesson_half_life_days", 30))
        self.reflector = Reflector(self.store, memory, generate, half_life_days=self.half_life,
                                   max_chars=int(config.mind("max_transcript_chars", 60000)))
        self.conv_dir = Path(config.mind("conversation_dir", "data/conversations"))
        self._present: set[str] = set()
        self._reminded: dict[str, float] = {}
        self._reflect_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        _current = self

    # ------------------------------------------------------------ lifecycle

    def start(self):
        if not self.store.get_state("initialized"):
            days = float(self.config.mind("backfill_days", 0))
            skipped = self.reflector.skip_old(self.conv_dir, days * 86400)
            self.store.set_state("initialized", str(int(time.time())))
            if skipped:
                logger.info(f"mind: first run, skipped {skipped} older transcripts (backfill_days={days:g})")
        self._tasks = [asyncio.create_task(self._reflect_loop()), asyncio.create_task(self._presence_loop())]
        logger.info("mind: reflection and planning started")

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _reflect_loop(self):
        await asyncio.sleep(float(self.config.mind("initial_delay_seconds", 120)))
        interval = max(60.0, float(self.config.mind("reflect_interval_minutes", 20)) * 60)
        while True:
            try:
                await self.reflect_pending()
            except Exception as e:
                logger.error(f"mind: reflection pass failed: {e}")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------ reflection

    async def reflect_pending(self, force: bool = False) -> list[dict]:
        async with self._reflect_lock:
            self.store.decay_lessons(self.half_life)
            pending = self.reflector.pending(
                self.conv_dir,
                quiet_seconds=float(self.config.mind("quiet_minutes", 10)) * 60,
                min_messages=int(self.config.mind("min_messages", 6)),
                force=force,
                gap_seconds=float(self.config.mind("session_gap_minutes", 20)) * 60,
            )
            limit = int(self.config.mind("max_sessions_per_pass", 5))
            results = []
            for group in pending[:limit]:
                try:
                    res = await self.reflector.reflect(group)
                except Exception as e:
                    logger.warning(f"mind: could not reflect on {group[0][0].name}: {e}")
                    continue
                results.append(res)
                logger.info(
                    f"mind: reflected on {res['session']}: {res['memories']} memories, {res['lessons']} lessons, "
                    f"{res['people']} people, {res['goals']} goals, {res['rejected']} rejected"
                )
            return results

    # -------------------------------------------------------------- presence

    def present_names(self) -> list[str]:
        mon = self.instance_monitor
        if mon is None:
            return []
        try:
            return [p.get("name") for p in mon.get_players() if p.get("name")]
        except Exception:
            return []

    async def _presence_loop(self):
        while True:
            try:
                await self.check_presence()
            except Exception as e:
                logger.debug(f"mind: presence check failed: {e}")
            await asyncio.sleep(15)

    async def check_presence(self):
        names = set(self.present_names())
        arrived = names - self._present
        self._present = names
        if not arrived:
            return
        cooldown = float(self.config.mind("reminder_cooldown_minutes", 30)) * 60
        for name in arrived:
            self.store.mark_seen(name)
            if not self.config.mind("commitment_reminders", True):
                continue
            items = self.store.open_items_for(name)
            if not items or time.time() - self._reminded.get(name.lower(), 0) < cooldown:
                continue
            self._reminded[name.lower()] = time.time()
            listed = "; ".join(f"#{g['id']} {g['title']}" for g in items[:3])
            text = (f"System update - {name} just joined the instance. You have open promises or plans with them: {listed}. "
                    f"Bring it up when it fits, and use the goals tool once it is done.")
            if self.session is not None:
                asyncio.create_task(self.session.send_text(text))

    # ------------------------------------------------------------------ hooks

    def prompt_block(self) -> str:
        return build_mind_block(
            self.store,
            lessons=int(self.config.mind("prompt_lessons", 8)),
            goals=int(self.config.mind("prompt_goals", 6)),
            half_life_days=self.half_life,
            present=self.present_names(),
        )

    def idle_prompt(self) -> str | None:
        if not self.config.mind("idle_goals", True):
            return None
        present = {n.lower() for n in self.present_names()}
        goals = self.store.active_goals(limit=20)
        promises = [g for g in goals if g["kind"] == "commitment" and (g.get("person") or "").lower() in present]
        own = [g for g in goals if g["kind"] in ("goal", "plan") and not g.get("person")]
        pick = promises[0] if promises else (random.choice(own[:3]) if own else None)
        if pick is None:
            return None
        if pick["kind"] == "commitment":
            return (f"System update - You have been idle for a while. {pick['person']} is here and you promised them: "
                    f"{pick['title']}. Bring it up or do it now.")
        return (f"System update - You have been idle for a while. Use the time on one of your goals: #{pick['id']} {pick['title']}"
                + (f" ({pick['detail']})" if pick.get("detail") else "")
                + ". Do something toward it, or talk to someone nearby about it, then note progress with the goals tool.")
