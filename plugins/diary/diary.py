"""Storage layer for the .diary file.

Custom plain text format, one file. Each entry is a header block followed
by a body, separated by blank lines. Easy for the AI to read whole, easy
for humans to open in notepad. Example:

    === 2026-05-08 part 3 (written 19:30:14) ===
    sessions: 5 (covered 14:22 - 19:18)
    people: HoppouAI, Archie, Sophie
    mood arc: started chill, got annoyed at the leprechaun thing, ended content

    body text in first person from gabriel's POV. multiple paragraphs ok.

    highlights:
    - played Canon Blast for Dylonall again
    - sophie changed VRChat name to neo-bestie
    === END ===

We never rewrite past entries, only ever append new ones. If something
needs correcting the AI can call updateDiary later (not implemented yet)
or the user can hand-edit the file, the parser is lenient.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# entry block markers. The bit inside (written ...) is just human-readable
# metadata, the parser captures it loosely so we can switch formats without
# breaking older diary files.
_HEADER_RE = re.compile(r"^===\s*(\d{4}-\d{2}-\d{2})\s+part\s+(\d+)\s*\(written\s+(.+?)\)\s*===\s*$")
_END_MARKER = "=== END ==="


@dataclass
class DiaryEntry:
    date: str                       # YYYY-MM-DD, used as the grouping key
    part: int                       # 1-based, multiple parts per day allowed
    written_at: str                 # human readable timestamp, US 12h format like "May 8, 2026 at 02:30:11 PM"
    sessions_covered: int = 0       # how many transcripts fed into this entry
    session_range: str = ""         # "hh:mm AM - hh:mm PM" of first/last session
    people: list[str] = field(default_factory=list)
    mood_arc: str = ""
    body: str = ""
    highlights: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render entry back to the on-disk text format."""
        lines = [f"=== {self.date} part {self.part} (written {self.written_at}) ==="]
        lines.append(f"sessions: {self.sessions_covered}" + (f" (covered {self.session_range})" if self.session_range else ""))
        if self.people:
            lines.append(f"people: {', '.join(self.people)}")
        if self.mood_arc:
            lines.append(f"mood arc: {self.mood_arc}")
        lines.append("")
        lines.append(self.body.strip() if self.body else "(no body)")
        if self.highlights:
            lines.append("")
            lines.append("highlights:")
            for h in self.highlights:
                lines.append(f"- {h}")
        lines.append(_END_MARKER)
        return "\n".join(lines)

    def header_summary(self) -> str:
        """Short one liner for list views."""
        bits = [f"{self.date} part {self.part}"]
        if self.session_range:
            bits.append(self.session_range)
        if self.people:
            bits.append(f"with {', '.join(self.people[:3])}" + (f" +{len(self.people)-3}" if len(self.people) > 3 else ""))
        return " | ".join(bits)


class DiaryStore:
    """Thread safe append-only diary on disk. Reads parse the whole file
    on demand, writes append a single entry block. The diary is small text
    and read rarely so a full reparse is fine."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_text(self) -> str:
        if not self.path.exists():
            return ""
        try:
            return self.path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"diary: failed to read {self.path}: {e}")
            return ""

    def append_entry(self, entry: DiaryEntry) -> None:
        rendered = entry.render()
        with self._lock:
            existing = self._read_text()
            sep = "\n\n" if existing.strip() else ""
            try:
                self.path.write_text(existing + sep + rendered + "\n", encoding="utf-8")
            except Exception as e:
                logger.error(f"diary: failed to write {self.path}: {e}")
                return
        logger.info(f"diary: appended {entry.date} part {entry.part} ({entry.sessions_covered} sessions)")

    def parse_all(self) -> list[DiaryEntry]:
        """Parse every entry from disk. Lenient, skips malformed blocks."""
        text = self._read_text()
        if not text.strip():
            return []
        entries: list[DiaryEntry] = []
        # split on END markers, then re-parse each chunk
        chunks = text.split(_END_MARKER)
        for raw in chunks:
            raw = raw.strip()
            if not raw:
                continue
            entry = self._parse_chunk(raw)
            if entry is not None:
                entries.append(entry)
        return entries

    def _parse_chunk(self, chunk: str) -> Optional[DiaryEntry]:
        lines = chunk.splitlines()
        # find header
        header_idx = None
        for i, ln in enumerate(lines):
            m = _HEADER_RE.match(ln.strip())
            if m:
                header_idx = i
                date, part, written = m.group(1), int(m.group(2)), m.group(3)
                break
        if header_idx is None:
            return None
        entry = DiaryEntry(date=date, part=part, written_at=written)
        # walk metadata lines until blank, then everything else is body until 'highlights:'
        i = header_idx + 1
        while i < len(lines) and lines[i].strip():
            ln = lines[i].strip()
            low = ln.lower()
            if low.startswith("sessions:"):
                rest = ln.split(":", 1)[1].strip()
                # try to extract count + range
                m = re.match(r"(\d+)(?:\s*\(covered\s+(.+?)\))?", rest)
                if m:
                    entry.sessions_covered = int(m.group(1))
                    entry.session_range = (m.group(2) or "").strip()
            elif low.startswith("people:"):
                rest = ln.split(":", 1)[1].strip()
                entry.people = [p.strip() for p in rest.split(",") if p.strip()]
            elif low.startswith("mood arc:"):
                entry.mood_arc = ln.split(":", 1)[1].strip()
            i += 1
        # skip blank
        while i < len(lines) and not lines[i].strip():
            i += 1
        # body until 'highlights:' line or end
        body_lines: list[str] = []
        while i < len(lines):
            if lines[i].strip().lower() == "highlights:":
                i += 1
                break
            body_lines.append(lines[i])
            i += 1
        entry.body = "\n".join(body_lines).strip()
        # highlights
        while i < len(lines):
            ln = lines[i].strip()
            if ln.startswith("- "):
                entry.highlights.append(ln[2:].strip())
            i += 1
        return entry

    def entries_for_date(self, date: str) -> list[DiaryEntry]:
        return [e for e in self.parse_all() if e.date == date]

    def latest_entry(self) -> Optional[DiaryEntry]:
        all_entries = self.parse_all()
        if not all_entries:
            return None
        # sort by date then part
        all_entries.sort(key=lambda e: (e.date, e.part))
        return all_entries[-1]

    def next_part_for(self, date: str) -> int:
        existing = self.entries_for_date(date)
        if not existing:
            return 1
        return max(e.part for e in existing) + 1

    def all_dates(self) -> list[str]:
        return sorted({e.date for e in self.parse_all()})

    def search(self, query: str, limit: int = 10) -> list[DiaryEntry]:
        """Cheap case insensitive substring search across body, highlights,
        people, mood arc. Sorted by date descending."""
        if not query:
            return []
        q = query.lower()
        hits: list[DiaryEntry] = []
        for e in self.parse_all():
            blob = " ".join([
                e.body or "",
                " ".join(e.highlights),
                " ".join(e.people),
                e.mood_arc or "",
            ]).lower()
            if q in blob:
                hits.append(e)
        hits.sort(key=lambda e: (e.date, e.part), reverse=True)
        return hits[:limit]


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def now_time_str() -> str:
    """US 12h format like "02:30:11 PM". Matches what the AI's time tool
    returns so the model doesnt have to mentally translate between 24h and
    12h when reading its own diary."""
    raw = datetime.now().strftime("%I:%M:%S %p")
    return raw.lstrip("0") if raw.startswith("0") else raw


def now_written_str() -> str:
    """Full friendly 'May 8, 2026 at 02:30:11 PM' style string for the
    diary header. Same format the AI sees from its time tool."""
    now = datetime.now()
    date_part = now.strftime("%B %d, %Y").replace(" 0", " ")  # strip leading zero on day
    time_part = now.strftime("%I:%M:%S %p")
    if time_part.startswith("0"):
        time_part = time_part[1:]
    return f"{date_part} at {time_part}"


def friendly_date(iso_date: str) -> str:
    """Turn '2026-05-08' into 'May 8, 2026'. Returns the input unchanged on
    parse failure so we never blow up on hand-edited dates."""
    try:
        d = datetime.strptime(iso_date, "%Y-%m-%d")
    except Exception:
        return iso_date
    return d.strftime("%B %d, %Y").replace(" 0", " ")
