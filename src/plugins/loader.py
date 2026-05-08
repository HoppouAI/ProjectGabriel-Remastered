"""Plugin discovery and lifecycle.

Walks `./plugins/`, reads each `plugin.yml` manifest, imports the entry
module, finds the `Plugin` subclass and calls `setup()`. Errors in any
single plugin are caught so the host stays up.
"""
import importlib.util
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.plugins.api import Plugin, PluginContext, emit_event

logger = logging.getLogger(__name__)

# Bump this when adding or changing anything in PluginContext that
# plugins can rely on. Plugins declare api_version in their manifest and
# we warn when they target a newer version than the host supports.
PLUGIN_API_VERSION = 1


# Per-plugin issue tally so the startup banner can show how many warnings
# or errors a plugin emitted while loading, and the actual message text
# for the first one. Filled by record_plugin_issue() and the
# _PluginLogCounter filter, read by src.cli._print_plugins_block.
_plugin_issues: dict[str, list[dict]] = {}
_PLUGIN_NAME_RE = re.compile(r"plugin '([^']+)'")
_REQUIRES_PREFIX_RE = re.compile(r"^plugin '[^']+'\s+", re.IGNORECASE)


def record_plugin_issue(plugin_name: str, level: str, message: str) -> None:
    """Direct insert into the per-plugin issue list. Use this from the
    loader when you want to tell the banner about a problem without
    making it scream in the console (the loader log itself can stay at
    DEBUG). Also writes to the daily plugins log file."""
    if not plugin_name:
        return
    _plugin_issues.setdefault(plugin_name, []).append({
        "level": level,
        "message": message,
    })
    _file_log = _get_file_logger()
    if _file_log is not None:
        line = f"[{plugin_name}] {message}"
        if level == "error":
            _file_log.error(line)
        else:
            _file_log.warning(line)


def get_plugin_issues() -> dict[str, list[dict]]:
    return {k: list(v) for k, v in _plugin_issues.items()}


def get_plugin_log_counts() -> dict[str, dict[str, int]]:
    """Backwards compat shim for older callers. Derives counts from the
    issue list."""
    out: dict[str, dict[str, int]] = {}
    for name, issues in _plugin_issues.items():
        d = {"warn": 0, "error": 0}
        for it in issues:
            if it["level"] == "error":
                d["error"] += 1
            else:
                d["warn"] += 1
        out[name] = d
    return out


class _PluginLogCounter(logging.Filter):
    """Tally WARNING/ERROR records emitted by plugins themselves
    (plugin.<name> namespace). Loader-side issues are recorded directly
    via record_plugin_issue, so we skip src.plugins.loader records here
    to avoid double counting."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        if not record.name.startswith("plugin."):
            return True
        name = record.name.split(".", 1)[1]
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        record_plugin_issue(
            name, "error" if record.levelno >= logging.ERROR else "warning", msg
        )
        return True


def _install_log_counter():
    # filters on a logger only fire when that logger's own .log() is
    # called; they do NOT run during propagation. so we attach to every
    # handler on the root logger instead, which is where StreamHandler
    # ends up after setup_logging().
    root = logging.getLogger()
    for h in root.handlers:
        if not any(isinstance(f, _PluginLogCounter) for f in h.filters):
            h.addFilter(_PluginLogCounter())
    if not any(isinstance(f, _PluginLogCounter) for f in root.filters):
        root.addFilter(_PluginLogCounter())


# Daily plugin log file. Holds WARNING+ records from anything under
# plugin.* namespace plus everything record_plugin_issue ever sees.
# Lives on its own logger so the file handler does not duplicate
# console output, and so we can keep the file at WARNING regardless of
# the root log level.
_plugins_log_path: Path | None = None
_file_logger: logging.Logger | None = None


def get_plugin_log_path() -> Path | None:
    return _plugins_log_path


def _get_file_logger() -> logging.Logger | None:
    return _file_logger


class _PluginNamespaceFilter(logging.Filter):
    """Pass only records from plugin.<name> loggers. Used so the file
    handler attached to root catches plugin internal warnings without
    swallowing the whole app."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("plugin.")


def _install_plugins_logfile():
    global _plugins_log_path, _file_logger
    if _plugins_log_path is not None:
        return
    log_dir = Path("data") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # date format requested: plugins-M-D-YYYY.log (no zero padding)
    now = datetime.now()
    fname = f"plugins-{now.month}-{now.day}-{now.year}.log"
    path = log_dir / fname
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    # internal logger for record_plugin_issue. propagate off so it does
    # not also show in the console handler.
    flog = logging.getLogger("src.plugins.file")
    flog.setLevel(logging.WARNING)
    flog.propagate = False
    flog.addHandler(fh)
    # also catch warnings emitted by plugins themselves via plugin.<name>.
    # we attach a sibling filtered handler to root so propagation reaches it.
    root_fh = logging.FileHandler(path, encoding="utf-8")
    root_fh.setLevel(logging.WARNING)
    root_fh.addFilter(_PluginNamespaceFilter())
    root_fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(root_fh)
    _plugins_log_path = path
    _file_logger = flog


class PluginManager:
    def __init__(self, config, plugins_dir: str = "plugins"):
        self.config = config
        self.plugins_dir = Path(plugins_dir)
        self.loaded: list[tuple[Plugin, PluginContext]] = []
        _install_log_counter()
        _install_plugins_logfile()

    def discover_and_load(self):
        """Scan the plugins dir, load each enabled plugin, call setup().
        Plugins that crash on load are logged and skipped, they will not
        take down the rest of the host."""
        if not self.plugins_dir.is_dir():
            logger.info(f"plugins dir '{self.plugins_dir}' missing, skipping plugin load")
            return

        # Global toggle. If config.plugins.enabled is false skip everything.
        if self.config is not None:
            if not bool(self.config.get("plugins", "enabled", default=True)):
                logger.info("plugins disabled in config, skipping plugin load")
                return

        for entry in sorted(self.plugins_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            manifest_path = entry / "plugin.yml"
            if not manifest_path.exists():
                logger.debug(f"no plugin.yml in {entry}, skipping")
                continue
            try:
                self._load_one(entry, manifest_path)
            except Exception as e:
                # crash-on-load is loud enough to keep at ERROR, also feed
                # the banner so the user sees it next to the plugin row.
                logger.error(f"failed to load plugin '{entry.name}': {e}", exc_info=True)
                record_plugin_issue(entry.name, "error", f"load failed: {e}")

        if self.loaded:
            names = ", ".join(p.name for p, _ in self.loaded)
            # banner already shows the full list, dont double print on console
            logger.debug(f"plugin manager: {len(self.loaded)} plugin(s) loaded -> {names}")
        else:
            logger.debug("plugin manager: no plugins loaded")

    def _load_one(self, plugin_dir: Path, manifest_path: Path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}

        name = manifest.get("name") or plugin_dir.name
        if not manifest.get("enabled", True):
            # quiet on console, banner will show the disabled dot anyway
            logger.debug(f"plugin '{name}' disabled in manifest, skipping")
            return

        # Per-plugin override. The "global" plugin enable is the manifest's
        # own `enabled:` field above. tools.yml only handles per-tool toggles
        # (under plugin_tools.<name>.<tool>) and is intentionally not
        # consulted here. The legacy host-config plugins.<name>.enabled
        # is still honored so old user configs keep working.
        if self.config is not None:
            cfg_enabled = self.config.get("plugins", name, "enabled", default=None)
            if cfg_enabled is False:
                logger.debug(f"plugin '{name}' disabled in host config, skipping")
                return

        api_version = int(manifest.get("api_version", 1))
        if api_version > PLUGIN_API_VERSION:
            msg = (
                f"targets api_version={api_version} but host only "
                f"supports up to {PLUGIN_API_VERSION}, may not behave"
            )
            logger.debug(f"plugin '{name}' {msg}")
            record_plugin_issue(name, "warning", msg)

        # Check declared python deps. Never auto pip install -- that
        # would be a footgun. Just warn so the user knows what to install.
        for req in manifest.get("requirements", []) or []:
            mod_name = req.split("==")[0].split(">=")[0].split("<")[0].split("~=")[0].strip()
            probe = mod_name.replace("-", "_")
            if importlib.util.find_spec(probe) is None:
                msg = f"requires '{req}' but it does not look installed."
                # full hint goes to plugins.log, banner shows the short msg
                logger.debug(f"plugin '{name}' {msg} run: pip install {req}")
                record_plugin_issue(name, "warning", msg)

        entry_file = manifest.get("entry", "__init__.py")
        entry_path = plugin_dir / entry_file
        if not entry_path.exists():
            msg = f"entry file '{entry_file}' not found"
            logger.error(f"plugin '{name}' {msg}")
            record_plugin_issue(name, "error", msg)
            return

        module_name = f"plugins.{plugin_dir.name}"
        spec = importlib.util.spec_from_file_location(
            module_name, entry_path, submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            logger.error(f"plugin '{name}' could not build import spec")
            record_plugin_issue(name, "error", "could not build import spec")
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"plugin '{name}' import failed: {e}", exc_info=True)
            record_plugin_issue(name, "error", f"import failed: {e}")
            return

        # Find the Plugin instance/class. Prefer an explicit `plugin`
        # attr, fall back to scanning module globals for a Plugin
        # subclass.
        plugin_obj: Plugin | None = None
        if hasattr(module, "plugin"):
            obj = getattr(module, "plugin")
            if isinstance(obj, Plugin):
                plugin_obj = obj
            elif isinstance(obj, type) and issubclass(obj, Plugin):
                plugin_obj = obj()
        if plugin_obj is None:
            for attr in vars(module).values():
                if isinstance(attr, type) and issubclass(attr, Plugin) and attr is not Plugin:
                    plugin_obj = attr()
                    break
        if plugin_obj is None:
            logger.error(f"plugin '{name}' has no Plugin subclass or `plugin` attribute")
            record_plugin_issue(name, "error", "has no Plugin subclass or `plugin` attribute")
            return

        # Let manifest fill in metadata when the class did not set it.
        if not getattr(plugin_obj, "name", None) or plugin_obj.name == "unnamed":
            plugin_obj.name = name
        if "version" in manifest:
            plugin_obj.version = str(manifest["version"])
        if "description" in manifest and not plugin_obj.description:
            plugin_obj.description = manifest["description"]
        if "author" in manifest and not plugin_obj.author:
            plugin_obj.author = manifest["author"]

        ctx = PluginContext(plugin_obj.name, self.config, plugin_dir)
        try:
            plugin_obj.setup(ctx)
        except Exception as e:
            logger.error(f"plugin '{plugin_obj.name}' setup() crashed: {e}", exc_info=True)
            record_plugin_issue(plugin_obj.name, "error", f"setup() crashed: {e}")
            return

        self.loaded.append((plugin_obj, ctx))
        # the banner shows version + author per plugin so we keep this DEBUG
        # to avoid duplicating that info in the startup log.
        logger.debug(
            f"loaded plugin '{plugin_obj.name}' v{plugin_obj.version}"
            + (f" by {plugin_obj.author}" if plugin_obj.author else "")
        )

    def bind_app(self, **refs):
        """Called by main() once audio/osc/session are constructed so
        plugin contexts can expose them."""
        for _, ctx in self.loaded:
            ctx.bind_app(**refs)

    def emit(self, event: str, *args, **kwargs):
        emit_event(event, *args, **kwargs)

    async def teardown_all(self):
        for plugin_obj, ctx in self.loaded:
            try:
                res = plugin_obj.teardown(ctx)
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:
                logger.error(
                    f"plugin '{plugin_obj.name}' teardown crashed: {e}", exc_info=True
                )
        self.loaded.clear()
