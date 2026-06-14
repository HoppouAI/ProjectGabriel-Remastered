"""Install / remove plugins and their pip dependencies.

Runs `bin/uv.exe pip install ...` against the active venv to grab
plugin requirements. uv auto detects `.venv/` so we just shell out and
let it work.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.plugin_installer.sources import PluginInfo, PluginSource


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_DIR = PROJECT_ROOT / "plugins"
UV_BIN = PROJECT_ROOT / "bin" / ("uv.exe" if sys.platform == "win32" else "uv")


@dataclass
class InstallResult:
    name: str
    ok: bool
    copied: bool = False
    deps_ok: bool = True
    skipped_deps: bool = False
    seeded_configs: int = 0   # files freshly created from a template
    merged_configs: int = 0   # existing files that gained new keys from the template
    error: Optional[str] = None
    log: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.log is None:
            self.log = []


# ---- example config seeding ----------------------------------------------


def _example_target_name(name: str) -> Optional[str]:
    """If `name` looks like an example template, return the real name to
    copy it to. Handles both conventions plugins use:

        foo.bar.example  -> foo.bar     (suffix style)
        foo.example.bar  -> foo.bar     (middle style)

    Returns None for files that don't match either pattern.
    """
    if name.endswith(".example"):
        stripped = name[: -len(".example")]
        return stripped or None
    if ".example." in name:
        head, _, tail = name.partition(".example.")
        if head and tail:
            return f"{head}.{tail}"
    return None


def seed_example_configs(
    plugin_dir: Path,
    on_line: Optional[Callable[[str], None]] = None,
) -> tuple[int, int]:
    """Bring every top level *.example template into sync with its
    real file. Two paths:

      - target missing  -> copy the template verbatim ("seed")
      - target exists   -> structural merge that adds any new keys
                           from the template into the user's file while
                           leaving every existing value (and comment)
                           in place ("merge"). YAML uses ruamel.yaml
                           round trip so comments survive. JSON gets a
                           plain deep merge.

    User values always win on conflicts, no key is ever removed.
    Returns (seeded_count, merged_count).
    """
    seeded = 0
    merged = 0
    if not plugin_dir.is_dir():
        return 0, 0
    for entry in sorted(plugin_dir.iterdir()):
        if not entry.is_file():
            continue
        target_name = _example_target_name(entry.name)
        if not target_name:
            continue
        target = plugin_dir / target_name
        if not target.exists():
            try:
                shutil.copy2(entry, target)
            except Exception as e:
                if on_line:
                    on_line(f"  failed to seed {target_name}: {e}")
                continue
            seeded += 1
            if on_line:
                on_line(f"  seeded {target_name} from {entry.name}")
            continue

        # Target already exists, try a key-add merge.
        ext = target.suffix.lower()
        try:
            if ext in (".yml", ".yaml"):
                added = _merge_yaml_keys(entry, target)
            elif ext == ".json":
                added = _merge_json_keys(entry, target)
            else:
                added = []
        except Exception as e:
            if on_line:
                on_line(f"  could not merge {target_name} ({e}), leaving it alone")
            continue

        if added:
            merged += 1
            preview = ", ".join(added[:5])
            if len(added) > 5:
                preview += f", +{len(added) - 5} more"
            if on_line:
                on_line(
                    f"  merged {target_name}: added {len(added)} new key(s) ({preview})"
                )
    return seeded, merged


def _merge_yaml_keys(template_path: Path, user_path: Path) -> list[str]:
    """Add keys present in `template_path` but missing from `user_path`
    to the user's file. Uses ruamel.yaml round trip mode so existing
    formatting and comments are preserved, and new keys get carried
    over with their template comments attached. Returns dotted key
    paths that were added."""
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True

    with open(template_path, "r", encoding="utf-8") as f:
        tmpl_data = yaml.load(f)
    with open(user_path, "r", encoding="utf-8") as f:
        user_data = yaml.load(f)

    if tmpl_data is None or user_data is None:
        return []
    if not _is_mapping(user_data) or not _is_mapping(tmpl_data):
        return []

    added: list[str] = []
    _yaml_add_missing(user_data, tmpl_data, "", added)
    if not added:
        return []

    with open(user_path, "w", encoding="utf-8") as f:
        yaml.dump(user_data, f)
    return added


def _is_mapping(obj: object) -> bool:
    # ruamel CommentedMap is a dict subclass, this still catches it.
    return isinstance(obj, dict)


def _yaml_add_missing(user_map, tmpl_map, prefix: str, added: list[str]) -> None:
    # ruamel stores "comment before key K" on the PREVIOUS key's ca.items
    # entry. The entry is a 4 slot list; the comment may be at slot 1
    # (list of CommentToken) or slot 2 (single CommentToken) depending
    # on the surrounding blank lines and indent. So when we lift a new
    # key over, we also lift any of those slots from the prior key's
    # entry that the user doesn't already have set.
    tmpl_keys = list(tmpl_map.keys())
    tmpl_ca = getattr(tmpl_map, "ca", None)
    user_ca = getattr(user_map, "ca", None)

    for idx, (key, value) in enumerate(tmpl_map.items()):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key in user_map:
            if _is_mapping(value) and _is_mapping(user_map.get(key)):
                _yaml_add_missing(user_map[key], value, path, added)
            continue

        user_map[key] = value

        if tmpl_ca is not None and user_ca is not None and hasattr(tmpl_ca, "items"):
            own = tmpl_ca.items.get(key)
            if own is not None:
                user_ca.items[key] = list(own)
            if idx > 0:
                prev_key = tmpl_keys[idx - 1]
                if prev_key in user_map:
                    tmpl_prev = tmpl_ca.items.get(prev_key)
                    if tmpl_prev is not None:
                        user_prev = user_ca.items.get(prev_key)
                        if user_prev is None:
                            user_ca.items[prev_key] = [
                                None,
                                tmpl_prev[1] if len(tmpl_prev) > 1 else None,
                                tmpl_prev[2] if len(tmpl_prev) > 2 else None,
                                tmpl_prev[3] if len(tmpl_prev) > 3 else None,
                            ]
                        else:
                            for slot in (1, 2, 3):
                                if (slot < len(tmpl_prev) and slot < len(user_prev)
                                        and tmpl_prev[slot] is not None
                                        and not user_prev[slot]):
                                    user_prev[slot] = tmpl_prev[slot]

        added.append(path)


def _merge_json_keys(template_path: Path, user_path: Path) -> list[str]:
    import json

    with open(template_path, "r", encoding="utf-8") as f:
        tmpl_data = json.load(f)
    with open(user_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    if not isinstance(user_data, dict) or not isinstance(tmpl_data, dict):
        return []

    added: list[str] = []
    _json_add_missing(user_data, tmpl_data, "", added)
    if not added:
        return []

    with open(user_path, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return added


def _json_add_missing(user_obj, tmpl_obj, prefix: str, added: list[str]) -> None:
    for key, value in tmpl_obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in user_obj:
            user_obj[key] = value
            added.append(path)
        elif isinstance(value, dict) and isinstance(user_obj.get(key), dict):
            _json_add_missing(user_obj[key], value, path, added)


# ---- pip install via uv --------------------------------------------------


def uv_available() -> bool:
    return UV_BIN.exists()


def install_requirements(
    reqs: list[str],
    on_line: Optional[Callable[[str], None]] = None,
) -> tuple[bool, list[str]]:
    """Run `uv pip install <reqs>`. Returns (ok, captured lines).

    If uv is missing we report that as a non-fatal warning (deps_ok =
    False) and let the caller decide what to do, the plugin folder is
    still copied either way.
    """
    if not reqs:
        return True, []
    if not UV_BIN.exists():
        msg = f"uv not found at {UV_BIN}, skipping pip install (run setup.bat)"
        if on_line:
            on_line(msg)
        return False, [msg]

    cmd = [str(UV_BIN), "pip", "install", *reqs]
    if on_line:
        on_line("$ " + " ".join(cmd))

    captured: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as e:
        msg = f"failed to launch uv: {e}"
        if on_line:
            on_line(msg)
        return False, [msg]

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        captured.append(line)
        if on_line:
            on_line(line)
    rc = proc.wait()
    return rc == 0, captured


# ---- install / remove ----------------------------------------------------


def install_plugin(
    info: PluginInfo,
    source: PluginSource,
    *,
    install_deps: bool = True,
    on_line: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """Materialize the plugin into ./plugins/ and pip install its reqs."""
    res = InstallResult(name=info.name, ok=False)
    log = res.log

    def emit(line: str) -> None:
        log.append(line)
        if on_line:
            on_line(line)

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    dest_existed = (PLUGINS_DIR / info.name).is_dir()

    emit(f"copying {info.name} v{info.version} from {info.source_label}")
    try:
        dest = source.materialize(info, PLUGINS_DIR)
    except Exception as e:
        res.error = f"copy failed: {e}"
        emit(res.error)
        return res
    res.copied = True
    emit(f"  -> {dest}")
    if dest_existed:
        emit("  updating in place, your local data and configs are kept")

    seeded, merged = seed_example_configs(dest, on_line=emit)
    res.seeded_configs = seeded
    res.merged_configs = merged

    if info.requirements:
        if install_deps:
            emit(f"installing {len(info.requirements)} requirement(s) via uv")
            ok, _ = install_requirements(info.requirements, on_line=on_line)
            res.deps_ok = ok
            if not ok:
                emit("  requirement install failed (plugin folder is still in place)")
        else:
            res.skipped_deps = True
            emit("skipped requirements per user choice")
    else:
        emit("no pip requirements")

    res.ok = res.copied and (res.deps_ok or res.skipped_deps)
    return res


def remove_plugin(name: str, on_line: Optional[Callable[[str], None]] = None) -> bool:
    """Delete plugins/<name>/. Does not touch installed pip deps because
    they may be shared with the host or other plugins."""
    target = PLUGINS_DIR / name
    if not target.exists():
        if on_line:
            on_line(f"plugin '{name}' is not installed")
        return False
    try:
        shutil.rmtree(target)
        if on_line:
            on_line(f"removed {target}")
        return True
    except Exception as e:
        if on_line:
            on_line(f"could not remove {target}: {e}")
        return False
