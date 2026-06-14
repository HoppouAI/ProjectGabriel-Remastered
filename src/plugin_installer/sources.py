"""Plugin source backends.

A "source" is anywhere we can pull plugin folders from. Two are built in:

- `GitHubSource` -- downloads a ZIP archive of the
  `HoppouAI/ProjectGabriel-Plugins` repo (or any fork) and indexes the
  top level folders that look like plugins. The zip is cached for the
  duration of the installer run.
- `LocalSource` -- points at a folder on disk and picks up every
  immediate subfolder that has a `plugin.yml`.

Both produce `PluginInfo` records with everything the TUI needs to
render the picker, and a `materialize(name, dest)` call that copies the
plugin folder to the destination.
"""
from __future__ import annotations

import io
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


DEFAULT_REPO = "HoppouAI/ProjectGabriel-Plugins"
DEFAULT_BRANCH = "main"


@dataclass
class PluginInfo:
    name: str
    version: str = "?"
    api_version: int = 1
    author: str = "unknown"
    description: str = ""
    requirements: list[str] = field(default_factory=list)
    enabled_default: bool = True
    # Where this entry came from, used for diagnostics / UI labels
    source_label: str = ""
    # Backend specific lookup key. For GitHub it's the inner zip prefix,
    # for local it's the absolute folder path.
    locator: str = ""

    @property
    def short_desc(self) -> str:
        d = self.description.strip().replace("\n", " ")
        return d if len(d) <= 80 else d[:77] + "..."


class SourceError(RuntimeError):
    pass


class PluginSource:
    """Base interface."""

    label: str = "source"

    def list_plugins(self) -> list[PluginInfo]:
        raise NotImplementedError

    def materialize(self, info: PluginInfo, dest_root: Path) -> Path:
        """Copy the plugin folder for `info` into `dest_root` and return
        the resulting plugin folder path. Overlays files on top of any
        existing install so user data (configs, caches, sqlite dbs,
        downloaded models, etc) is preserved across upgrades.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any cached resources (temp files, zip handles)."""
        pass


# ---- GitHub --------------------------------------------------------------


class GitHubSource(PluginSource):
    """Pulls a zip of the plugins repo and reads it in memory.

    The zip is downloaded once on first `list_plugins()` call and
    cached. `materialize()` reuses the same zip handle.
    """

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        branch: str = DEFAULT_BRANCH,
        progress: Optional[callable] = None,
    ):
        self.repo = repo
        self.branch = branch
        self.label = f"GitHub: {repo}@{branch}"
        self._zip_bytes: Optional[bytes] = None
        self._zip: Optional[zipfile.ZipFile] = None
        self._inner_root: Optional[str] = None  # e.g. ProjectGabriel-Plugins-main/
        self._progress = progress

    @property
    def archive_url(self) -> str:
        return f"https://github.com/{self.repo}/archive/refs/heads/{self.branch}.zip"

    def _ensure_loaded(self) -> None:
        if self._zip is not None:
            return
        if self._progress:
            self._progress(f"downloading {self.archive_url} ...")
        try:
            req = urllib.request.Request(
                self.archive_url,
                headers={"User-Agent": "ProjectGabriel-PluginInstaller"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                self._zip_bytes = resp.read()
        except urllib.error.URLError as e:
            raise SourceError(f"could not reach github: {e}") from e
        except Exception as e:
            raise SourceError(f"download failed: {e}") from e

        try:
            self._zip = zipfile.ZipFile(io.BytesIO(self._zip_bytes))
        except zipfile.BadZipFile as e:
            raise SourceError(f"github returned a malformed zip: {e}") from e

        # the archive root is always one folder, e.g. ProjectGabriel-Plugins-main/
        names = self._zip.namelist()
        if not names:
            raise SourceError("github zip was empty")
        self._inner_root = names[0].split("/", 1)[0] + "/"

    def list_plugins(self) -> list[PluginInfo]:
        self._ensure_loaded()
        assert self._zip is not None and self._inner_root is not None

        # find every <root>/<plugin>/plugin.yml
        manifests: dict[str, str] = {}
        for name in self._zip.namelist():
            if not name.startswith(self._inner_root):
                continue
            rel = name[len(self._inner_root):]
            parts = rel.split("/", 2)
            if len(parts) >= 2 and parts[1] == "plugin.yml":
                manifests[parts[0]] = name

        out: list[PluginInfo] = []
        for plugin_dir, manifest_path in sorted(manifests.items()):
            try:
                with self._zip.open(manifest_path) as f:
                    data = yaml.safe_load(f.read().decode("utf-8")) or {}
            except Exception:
                continue
            info = _info_from_manifest(data, fallback_name=plugin_dir)
            info.source_label = self.label
            info.locator = self._inner_root + plugin_dir + "/"
            out.append(info)
        return out

    def materialize(self, info: PluginInfo, dest_root: Path) -> Path:
        # Overlays the new files on top of any existing install instead
        # of wiping the folder, so user data (configs, caches, sqlite
        # dbs, downloaded models, etc) survives an upgrade.
        self._ensure_loaded()
        assert self._zip is not None
        prefix = info.locator
        if not prefix.endswith("/"):
            prefix += "/"
        dest_dir = dest_root / info.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for name in self._zip.namelist():
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix):]
            if not rel:
                continue
            target = dest_dir / rel
            if name.endswith("/"):
                if target.exists() and not target.is_dir():
                    target.unlink()
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_dir():
                shutil.rmtree(target)
            with self._zip.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return dest_dir

    def close(self) -> None:
        if self._zip is not None:
            try:
                self._zip.close()
            except Exception:
                pass
        self._zip = None
        self._zip_bytes = None


# ---- local folder --------------------------------------------------------


_LOCAL_IGNORE_NAMES = {"__pycache__", ".git", ".idea", ".vscode", "node_modules"}


class LocalSource(PluginSource):
    """Reads plugins out of a folder on disk. The folder can either:

    - contain plugin folders directly (root/plugin_a/plugin.yml), OR
    - itself be a single plugin folder (root/plugin.yml)

    Both layouts are handled.
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.label = f"Local: {self.root}"
        self._single_plugin = (self.root / "plugin.yml").exists()

    def list_plugins(self) -> list[PluginInfo]:
        if not self.root.exists():
            raise SourceError(f"path not found: {self.root}")

        if self._single_plugin:
            entries = [self.root]
        else:
            if not self.root.is_dir():
                raise SourceError(f"not a directory: {self.root}")
            entries = sorted(
                p for p in self.root.iterdir()
                if p.is_dir() and not p.name.startswith((".", "_"))
                and (p / "plugin.yml").exists()
            )

        out: list[PluginInfo] = []
        for entry in entries:
            try:
                with open(entry / "plugin.yml", "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                continue
            info = _info_from_manifest(data, fallback_name=entry.name)
            info.source_label = self.label
            info.locator = str(entry)
            out.append(info)

        if not out:
            raise SourceError(f"no plugins (with plugin.yml) found in {self.root}")
        return out

    def materialize(self, info: PluginInfo, dest_root: Path) -> Path:
        # Overlay onto any existing install rather than wiping. Same
        # rationale as GitHubSource: keep the user's data and configs.
        src = Path(info.locator)
        if not src.is_dir():
            raise SourceError(f"source folder vanished: {src}")
        dest = dest_root / info.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, ignore=_local_ignore, dirs_exist_ok=True)
        return dest


def _local_ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _LOCAL_IGNORE_NAMES or n.endswith(".pyc")}


# ---- shared helpers ------------------------------------------------------


def _info_from_manifest(data: dict, fallback_name: str) -> PluginInfo:
    reqs = data.get("requirements") or []
    if not isinstance(reqs, list):
        reqs = []
    return PluginInfo(
        name=str(data.get("name") or fallback_name),
        version=str(data.get("version") or "?"),
        api_version=int(data.get("api_version") or 1),
        author=str(data.get("author") or "unknown"),
        description=str(data.get("description") or ""),
        requirements=[str(r) for r in reqs if r],
        enabled_default=bool(data.get("enabled", True)),
    )


def read_installed(plugins_dir: Path) -> dict[str, PluginInfo]:
    """Return {name: PluginInfo} for plugins already in `plugins_dir`."""
    out: dict[str, PluginInfo] = {}
    if not plugins_dir.is_dir():
        return out
    for entry in sorted(plugins_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        manifest = entry / "plugin.yml"
        if not manifest.exists():
            continue
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            continue
        info = _info_from_manifest(data, fallback_name=entry.name)
        info.source_label = "installed"
        info.locator = str(entry)
        out[info.name] = info
    return out
