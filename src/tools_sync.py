"""Syncs the live tool registry into config/tools.yml on startup.

We walk every @register_tool class plus the emotion declarations and
plugin folders, then make sure each tool name appears in tools.yml so
the operator can flip it on/off in the configurator. New stuff defaults
to enabled so the user never silently loses a tool after an update.

The file is split into two top level sections:

    tools:           # built-in tools that ship with the host
      <name>: bool
    plugin_tools:    # modular plugin tools, grouped per plugin
      <plugin_name>:
        <name>: bool

Whether the plugin itself is loaded is governed by the plugin's own
plugin.yml `enabled:` flag, NOT by anything in tools.yml. The toggles
under plugin_tools.<plugin> only control which of that plugin's tools
get exposed to gemini once the plugin is loaded.

This file gets touched at startup AFTER plugins have been discovered and
loaded so plugin tools are in the registry by the time we sync.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from src.tools._base import get_registered_tools

logger = logging.getLogger(__name__)

TOOLS_PATH = Path("config/tools.yml")

HEADER_COMMENT = """\
config/tools.yml -- per-tool / per-plugin-tool toggles

This file is auto-managed. On every startup ProjectGabriel walks the
live tool registry and adds any newly discovered tool here, defaulting
it to true. Existing values are never overwritten, so anything you flip
off here stays off across upgrades.

Schema:
  tools:                 # built-in tools shipped with the host
    <tool_name>: bool
  plugin_tools:          # tools added by modular plugins, grouped per plugin
    <plugin_name>:
      <tool_name>: bool

How disabling works:
  set a tool to `false` and its FunctionDeclaration is filtered out of
  the schema sent to gemini on connect, so the model has no idea the
  tool exists and cannot call it. The python handler still lives in
  memory but is unreachable.

Plugin on/off:
  whether a plugin itself loads is controlled by `enabled:` inside that
  plugin's own plugins/<name>/plugin.yml, NOT by anything here. The
  toggles under plugin_tools.<plugin> only hide individual tools once
  the plugin is loaded.

Examples:
  tools:
    vrchatJump: false        # disables a single built-in tool
  plugin_tools:
    suno:
      generateSong: false    # plugin still loads, but this tool is hidden
"""


class _PermissiveCfg:
    """A stand in config that says yes to everything.

    Lots of tools self-gate inside `declarations()` based on whether some
    config flag is on (discord, social, suno, vrchat api, etc). For the
    purpose of enumerating what tools EXIST so we can list them in
    tools.yml we want them all to show up. So we hand them this stub
    instead of the real Config.
    """

    def get(self, *_keys, default=None):
        if default is None or default is False:
            return True
        return default

    def __getattr__(self, _name):
        # bool-y truthy for any attribute lookup
        return True


def _emotion_decl_names(real_config) -> Iterable[str]:
    # emotions are config driven (each animation becomes a callable name),
    # so we use the real Config here. with no config we just skip them.
    if real_config is None:
        return []
    try:
        from src.emotions import generate_emotion_function_declarations
    except Exception:
        return []
    try:
        decls = generate_emotion_function_declarations(real_config) or []
    except Exception:
        return []
    out = []
    for d in decls:
        if isinstance(d, dict):
            out.append(d.get("name"))
        else:
            out.append(getattr(d, "name", None))
    return [n for n in out if n]


def _collect_registry(real_config=None) -> tuple[set[str], dict[str, set[str]]]:
    """Returns (builtin_tool_names, {plugin_name: {tool_names}})."""
    stub = _PermissiveCfg()
    builtin: set[str] = set()
    plugin_tools: dict[str, set[str]] = {}

    for cls in get_registered_tools():
        module = getattr(cls, "__module__", "") or ""
        try:
            instance = cls.__new__(cls)
            instance.handler = None
            decls = instance.declarations(config=stub) or []
        except Exception as exc:
            logger.debug(f"declarations() failed for {cls.__name__}: {exc}")
            decls = []
        names = [getattr(d, "name", None) for d in decls]
        names = [n for n in names if n]
        if module.startswith("plugins."):
            pname = module.split(".", 2)[1] if "." in module else cls.__name__
            plugin_tools.setdefault(pname, set()).update(names)
        else:
            builtin.update(names)

    builtin.update(_emotion_decl_names(real_config))
    return builtin, plugin_tools


def sync_tools_yml(real_config=None) -> dict:
    """Reconcile config/tools.yml with whatever's currently registered.

    - adds any newly discovered tool/plugin entry, defaulting to enabled
    - leaves existing user values alone (never flips on -> off or vice versa)
    - does NOT prune stale entries (operator may toggle them back later
      after an upgrade, and pruning makes the diff noisy)
    - preserves yaml comments via ruamel
    """
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.indent(mapping=2, sequence=4, offset=2)

    if TOOLS_PATH.exists():
        try:
            with open(TOOLS_PATH, "r", encoding="utf-8") as f:
                data = yaml_rt.load(f)
        except Exception as exc:
            logger.error(f"could not parse {TOOLS_PATH}, recreating: {exc}")
            data = None
    else:
        data = None

    if not isinstance(data, CommentedMap) and not isinstance(data, dict):
        data = CommentedMap()

    if "tools" not in data or not isinstance(data.get("tools"), dict):
        data["tools"] = CommentedMap()
    if "plugin_tools" not in data or not isinstance(data.get("plugin_tools"), dict):
        data["plugin_tools"] = CommentedMap()

    builtin, plugin_tools = _collect_registry(real_config)

    added_tools = 0
    added_plugin_tools = 0

    tools_block = data["tools"]
    for name in sorted(builtin):
        if name not in tools_block:
            tools_block[name] = True
            added_tools += 1

    pt_block = data["plugin_tools"]
    for pname in sorted(plugin_tools.keys()):
        if pname not in pt_block or not isinstance(pt_block.get(pname), dict):
            pt_block[pname] = CommentedMap()
        sub = pt_block[pname]
        for name in sorted(plugin_tools[pname]):
            if name not in sub:
                sub[name] = True
                added_plugin_tools += 1

    # always rewrite if file missing, otherwise only if we actually added stuff
    file_missing = not TOOLS_PATH.exists()

    # Stamp the header comment on if it isn't already there. ruamel uses
    # the start comment as the file banner. We detect by checking the
    # CommentToken on the top mapping.
    needs_header = True
    try:
        existing_comment = data.ca.comment if hasattr(data, "ca") else None
        if existing_comment and existing_comment[1]:
            for tok in existing_comment[1]:
                if "auto-managed" in str(getattr(tok, "value", "")):
                    needs_header = False
                    break
    except Exception:
        pass
    if needs_header and isinstance(data, CommentedMap):
        data.yaml_set_start_comment(HEADER_COMMENT)

    if file_missing or added_tools or added_plugin_tools or needs_header:
        TOOLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TOOLS_PATH, "w", encoding="utf-8") as f:
            yaml_rt.dump(data, f)
        if file_missing:
            logger.info(
                f"tools.yml created with {len(builtin)} tools and "
                f"{sum(len(v) for v in plugin_tools.values())} plugin tools"
            )
        else:
            logger.info(
                f"tools.yml updated: +{added_tools} tools, +{added_plugin_tools} plugin tools"
            )

    return {
        "added_tools": added_tools,
        "added_plugin_tools": added_plugin_tools,
        "tools_total": len(builtin),
        "plugin_tools_total": sum(len(v) for v in plugin_tools.values()),
    }
