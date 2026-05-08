"""Plugin discovery and lifecycle.

Walks `./plugins/`, reads each `plugin.yml` manifest, imports the entry
module, finds the `Plugin` subclass and calls `setup()`. Errors in any
single plugin are caught so the host stays up.
"""
import importlib.util
import logging
import sys
from pathlib import Path

import yaml

from src.plugins.api import Plugin, PluginContext, emit_event

logger = logging.getLogger(__name__)

# Bump this when adding or changing anything in PluginContext that
# plugins can rely on. Plugins declare api_version in their manifest and
# we warn when they target a newer version than the host supports.
PLUGIN_API_VERSION = 1


class PluginManager:
    def __init__(self, config, plugins_dir: str = "plugins"):
        self.config = config
        self.plugins_dir = Path(plugins_dir)
        self.loaded: list[tuple[Plugin, PluginContext]] = []

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
                logger.error(f"failed to load plugin '{entry.name}': {e}", exc_info=True)

        if self.loaded:
            names = ", ".join(p.name for p, _ in self.loaded)
            logger.info(f"plugin manager: {len(self.loaded)} plugin(s) loaded -> {names}")
        else:
            logger.info("plugin manager: no plugins loaded")

    def _load_one(self, plugin_dir: Path, manifest_path: Path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}

        name = manifest.get("name") or plugin_dir.name
        if not manifest.get("enabled", True):
            logger.info(f"plugin '{name}' disabled in manifest, skipping")
            return

        # Per-plugin override from the host config
        # (config.yml -> plugins -> <name> -> enabled: false)
        if self.config is not None:
            cfg_enabled = self.config.get("plugins", name, "enabled", default=None)
            if cfg_enabled is False:
                logger.info(f"plugin '{name}' disabled in host config, skipping")
                return

        api_version = int(manifest.get("api_version", 1))
        if api_version > PLUGIN_API_VERSION:
            logger.warning(
                f"plugin '{name}' targets api_version={api_version} but host "
                f"only supports up to {PLUGIN_API_VERSION}, may not behave"
            )

        # Check declared python deps. Never auto pip install -- that
        # would be a footgun. Just warn so the user knows what to install.
        for req in manifest.get("requirements", []) or []:
            mod_name = req.split("==")[0].split(">=")[0].split("<")[0].split("~=")[0].strip()
            probe = mod_name.replace("-", "_")
            if importlib.util.find_spec(probe) is None:
                logger.warning(
                    f"plugin '{name}' requires '{req}' but it does not look installed. "
                    f"run: pip install {req}"
                )

        entry_file = manifest.get("entry", "__init__.py")
        entry_path = plugin_dir / entry_file
        if not entry_path.exists():
            logger.error(f"plugin '{name}' entry file '{entry_file}' not found")
            return

        module_name = f"plugins.{plugin_dir.name}"
        spec = importlib.util.spec_from_file_location(
            module_name, entry_path, submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            logger.error(f"plugin '{name}' could not build import spec")
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"plugin '{name}' import failed: {e}", exc_info=True)
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
            return

        self.loaded.append((plugin_obj, ctx))
        logger.info(
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
