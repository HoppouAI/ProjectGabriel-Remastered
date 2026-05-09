"""Diary plugin entry.

Spins up a DiaryStore + DiaryScheduler, registers four tools (read, search,
list, force-update). Background scheduler ticks every couple hours and feeds
the most recent VRChat session transcripts through gemini-3.1-flash-lite-preview
to produce a first person diary entry, appended to `data/plugins/diary/gabriel.diary`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.plugins import Plugin, PluginContext

from .diary import DiaryStore
from .scheduler import DiaryScheduler
from .summarizer import DEFAULT_MODEL
from .tools import DiaryTools

logger = logging.getLogger(__name__)

# fallback location, must match src.gemini_live.conversation_logger.CONVERSATION_DIR
_DEFAULT_CONV_DIR = Path("data/conversations")
_DEFAULT_INTERVAL_HOURS = 2.0
_DEFAULT_MAX_SESSIONS = 5
_DEFAULT_INITIAL_DELAY_SECONDS = 300.0


class DiaryPlugin(Plugin):
    name = "diary"
    version = "1.0.0"
    description = "Background diary writer + read tools, summarizes recent VRChat sessions into a long term first person diary."
    author = "HoppouAI"

    def setup(self, ctx: PluginContext):
        # config knobs all live under plugins.diary.* in config.yml
        interval_hours = float(ctx.plugin_config("interval_hours", _DEFAULT_INTERVAL_HOURS) or _DEFAULT_INTERVAL_HOURS)
        max_sessions = int(ctx.plugin_config("max_sessions", _DEFAULT_MAX_SESSIONS) or _DEFAULT_MAX_SESSIONS)
        model = str(ctx.plugin_config("model", DEFAULT_MODEL) or DEFAULT_MODEL)
        initial_delay = float(ctx.plugin_config("initial_delay_seconds", _DEFAULT_INITIAL_DELAY_SECONDS) or _DEFAULT_INITIAL_DELAY_SECONDS)
        diary_filename = str(ctx.plugin_config("filename", "gabriel.diary") or "gabriel.diary")
        conv_dir_str = ctx.plugin_config("conversation_dir")
        conv_dir = Path(conv_dir_str) if conv_dir_str else _DEFAULT_CONV_DIR

        store = DiaryStore(ctx.data_dir() / diary_filename)
        # ToolHandler instantiates with cls(handler) so we cant pass the store
        # via __init__, stash it as a class attr the same way mood does.
        DiaryTools._store = store

        def _resolve_persona() -> str:
            """Pull the active base persona from prompts.yml fresh each tick,
            so prompt edits take effect without restarting. Mirrors the lookup
            in Config.build_system_instruction() but skips appends/memories,
            the diary only wants the raw character voice."""
            cfg = ctx.config
            if cfg is None:
                return ""
            try:
                prompt_name = cfg.get("gemini", "prompt", default="normal")
                raw = (getattr(cfg, "_prompts", {}) or {}).get(prompt_name, "")
                if isinstance(raw, dict):
                    return str(raw.get("prompt", "")).strip()
                return str(raw or "").strip()
            except Exception as e:
                ctx.logger.warning(f"diary: failed to resolve persona: {e}")
                return ""

        scheduler = DiaryScheduler(
            store=store,
            conv_dir=conv_dir,
            get_api_key=lambda: getattr(ctx.config, "api_key", "") or "",
            interval_seconds=interval_hours * 3600,
            max_sessions=max_sessions,
            model=model,
            initial_delay_seconds=initial_delay,
            get_persona=_resolve_persona,
        )
        DiaryTools._scheduler = scheduler
        ctx.register_tool(DiaryTools)

        # start the background loop once the host's asyncio loop is up
        def _on_startup():
            scheduler.start()

        ctx.subscribe("startup", _on_startup)
        ctx.subscribe("shutdown", lambda: scheduler.stop())

        # keep handles for debug pokes
        self._store = store
        self._scheduler = scheduler

        ctx.logger.info(
            f"diary plugin ready, interval={interval_hours}h, max_sessions={max_sessions}, "
            f"model={model}, file={store.path}"
        )

    def teardown(self, ctx: PluginContext):
        try:
            if hasattr(self, "_scheduler") and self._scheduler is not None:
                self._scheduler.stop()
        except Exception as e:
            ctx.logger.warning(f"diary teardown failed: {e}")


plugin = DiaryPlugin
