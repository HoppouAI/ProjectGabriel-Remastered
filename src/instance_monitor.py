import re
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from glob import glob

logger = logging.getLogger(__name__)

VRCHAT_LOG_DIR = Path(os.environ.get("LOCALAPPDATA", "")).parent / "LocalLow" / "VRChat" / "VRChat"

_JOIN_PATTERN = re.compile(
    r"\[Behaviour\] OnPlayerJoined\s+(.+?)\s+\((usr_[0-9a-f\-]+)\)"
)
_LEAVE_PATTERN = re.compile(
    r"\[Behaviour\] OnPlayerLeft\s+(.+?)\s+\((usr_[0-9a-f\-]+)\)"
)
_JOINING_PATTERN = re.compile(
    r"\[Behaviour\] Joining\s+(wrld_[0-9a-f\-]+):(.+)"
)


def _find_latest_log() -> Path | None:
    if not VRCHAT_LOG_DIR.exists():
        return None
    logs = sorted(
        VRCHAT_LOG_DIR.glob("output_log_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


class InstanceMonitor:
    def __init__(self):
        self._players: dict[str, dict] = {}  # user_id -> {name, id, join_time}
        self._world_id: str = ""
        self._instance_id: str = ""
        self._log_path: Path | None = None
        self._file_pos: int = 0
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def current_location(self) -> str:
        if self._world_id and self._instance_id:
            return f"{self._world_id}:{self._instance_id}"
        return ""

    @property
    def player_count(self) -> int:
        return len(self._players)

    def get_players(self) -> list[dict]:
        return list(self._players.values())

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Instance monitor started")

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Instance monitor stopped")

    def _init_log(self):
        log_path = _find_latest_log()
        if log_path != self._log_path:
            self._log_path = log_path
            self._file_pos = 0
            if log_path:
                logger.info(f"Monitoring VRChat log: {log_path.name}")
                self._do_initial_scan()
            else:
                logger.warning("No VRChat log file found")

    def _do_initial_scan(self):
        if not self._log_path or not self._log_path.exists():
            return
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                self._file_pos = len(content.encode("utf-8", errors="replace"))

            last_join_pos = content.rfind("[Behaviour] Joining ")
            if last_join_pos == -1:
                return

            relevant = content[last_join_pos:]
            self._players.clear()
            self._world_id = ""
            self._instance_id = ""
            self._parse_chunk(relevant)
            logger.info(
                f"Initial scan: {len(self._players)} players in {self.current_location}"
            )
        except Exception as e:
            logger.error(f"Initial log scan failed: {e}")

    def _parse_chunk(self, chunk: str):
        for line in chunk.splitlines():
            m = _JOINING_PATTERN.search(line)
            if m:
                self._world_id = m.group(1)
                self._instance_id = m.group(2)
                self._players.clear()
                logger.debug(f"Joined instance: {self.current_location}")
                continue

            m = _JOIN_PATTERN.search(line)
            if m:
                display_name = m.group(1)
                user_id = m.group(2)
                self._players[user_id] = {
                    "name": display_name,
                    "id": user_id,
                    "join_time": datetime.now().isoformat(),
                }
                logger.debug(f"Player joined: {display_name} ({user_id})")
                continue

            m = _LEAVE_PATTERN.search(line)
            if m:
                user_id = m.group(2)
                left = self._players.pop(user_id, None)
                if left:
                    logger.debug(f"Player left: {left['name']} ({user_id})")

    def _read_new_lines(self):
        if not self._log_path or not self._log_path.exists():
            return

        try:
            file_size = self._log_path.stat().st_size
            if file_size < self._file_pos:
                self._file_pos = 0
                self._players.clear()

            if file_size == self._file_pos:
                return

            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                new_bytes = content.encode("utf-8", errors="replace")

            if len(new_bytes) <= self._file_pos:
                return

            new_content = new_bytes[self._file_pos:].decode("utf-8", errors="replace")
            self._file_pos = len(new_bytes)
            self._parse_chunk(new_content)
        except Exception as e:
            logger.error(f"Log read failed: {e}")

    async def _poll_loop(self):
        self._init_log()
        while self._running:
            try:
                await asyncio.to_thread(self._check_for_updates)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor poll error: {e}")
            await asyncio.sleep(2)

    def _check_for_updates(self):
        new_log = _find_latest_log()
        if new_log and new_log != self._log_path:
            self._log_path = new_log
            self._file_pos = 0
            self._players.clear()
            logger.info(f"Switched to new VRChat log: {new_log.name}")
            self._do_initial_scan()
        else:
            self._read_new_lines()
