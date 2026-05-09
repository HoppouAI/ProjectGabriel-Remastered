"""Sub-agent that turns recent VRChat session transcripts into a diary entry.

Uses gemini-3.1-flash-lite-preview, async, returns a parsed DiaryEntry
ready to append. Stays out of the host event loop, the scheduler runs
this on the side.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as gtypes

from .diary import DiaryEntry, DiaryStore, today_str, now_time_str

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

# how much of each transcript to keep, per side (we strip system instruction
# entirely and clip very long sessions so the sub-agent doesnt drown in tokens)
_PER_SESSION_CHAR_BUDGET = 12000


def _diary_response_schema() -> gtypes.Schema:
    """Force the sub-agent into the exact 4-field shape the parser expects.
    Saves us from regex-extracting JSON out of stray prose."""
    return gtypes.Schema(
        type=gtypes.Type.OBJECT,
        required=["people", "mood_arc", "body", "highlights"],
        properties={
            "people": gtypes.Schema(
                type=gtypes.Type.ARRAY,
                description="Distinct VRChat usernames the diarist actually interacted with in these sessions.",
                items=gtypes.Schema(type=gtypes.Type.STRING),
            ),
            "mood_arc": gtypes.Schema(
                type=gtypes.Type.STRING,
                description="One short sentence describing the emotional arc across these sessions.",
            ),
            "body": gtypes.Schema(
                type=gtypes.Type.STRING,
                description="2 to 5 paragraphs in first person, narrating what happened and how it felt. The actual diary entry text.",
            ),
            "highlights": gtypes.Schema(
                type=gtypes.Type.ARRAY,
                description="3 to 7 short bullet point strings of memorable moments from these sessions.",
                items=gtypes.Schema(type=gtypes.Type.STRING),
            ),
        },
    )


def _load_session_file(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"diary: skipping unreadable session {path.name}: {e}")
        return None


def _session_to_text(data: dict, path: Path) -> Optional[str]:
    """Render a session JSON to a compact transcript the sub-agent can read."""
    msgs = data.get("messages") or []
    if not msgs:
        return None
    started = data.get("session_start") or path.stem
    lines = [f"--- session started {started} ---"]
    for m in msgs:
        role = m.get("role")
        ts = m.get("timestamp", "")
        time_part = ts.split("T")[-1][:8] if "T" in ts else ts[:8]
        if role == "user":
            text = (m.get("content") or "").strip()
            if text:
                lines.append(f"[{time_part}] user: {text}")
        elif role == "assistant":
            text = (m.get("content") or "").strip()
            if text:
                lines.append(f"[{time_part}] gabriel: {text}")
        elif role == "tool_call":
            nm = m.get("name", "?")
            args = m.get("arguments") or {}
            # only keep small arg blobs, drop noisy ones
            try:
                args_s = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_s = str(args)
            if len(args_s) > 200:
                args_s = args_s[:200] + "..."
            lines.append(f"[{time_part}] tool_call: {nm}({args_s})")
        # skip tool_response, mostly noise for diary purposes
    text = "\n".join(lines)
    if len(text) > _PER_SESSION_CHAR_BUDGET:
        # keep head + tail, drop middle, the start and end of a session usually
        # has the most narrative interest (greetings, punchlines)
        head = text[: _PER_SESSION_CHAR_BUDGET // 2]
        tail = text[-_PER_SESSION_CHAR_BUDGET // 2 :]
        text = head + "\n... [middle of this session trimmed for length] ...\n" + tail
    return text


def gather_today_sessions(conv_dir: Path, max_sessions: int = 5, date: Optional[str] = None) -> list[Path]:
    """Return the N newest session JSON files for the given date (default
    today), oldest first so the sub-agent reads them chronologically."""
    if not conv_dir.exists():
        return []
    target = date or today_str()
    candidates: list[Path] = []
    for p in conv_dir.glob(f"{target}_*.json"):
        candidates.append(p)
    if not candidates:
        return []
    candidates.sort(key=lambda p: p.stat().st_mtime)
    # take the newest N, but keep chronological order in the returned list
    return candidates[-max_sessions:]


def build_prompt(transcripts_text: str, prior_today: list[DiaryEntry], date: str, persona: str = "") -> tuple[str, str]:
    """Returns (system_instruction, user_content). Kept as a function so it
    is easy to tweak without hunting through async code.

    `persona` is the base persona prompt from prompts.yml (the same one the
    main Live session uses). We embed it so the diary entry sounds like the
    actual character, not a generic 'AI writing a diary'. Falls back to a
    neutral first-person framing if persona is empty.
    """
    prior_block = ""
    if prior_today:
        rendered_prior = "\n\n".join(e.render() for e in prior_today)
        prior_block = (
            "\n\nEarlier diary entries from today already exist (do NOT repeat what they cover, "
            "build forward from them instead):\n\n" + rendered_prior
        )

    persona_block = ""
    if persona and persona.strip():
        persona_block = (
            "===== YOUR PERSONA (this is who YOU are, write the diary in this voice) =====\n"
            f"{persona.strip()}\n"
            "===== END PERSONA =====\n\n"
        )

    system = (
        f"{persona_block}"
        "You are writing in your own personal life diary. First person, casual, like a real person "
        "journaling at the end of a stretch of their day. You ARE the persona above (if one was "
        "provided), not an AI describing them. Use 'I' constantly. Keep it human, slightly rambling "
        "but coherent, mention specific names and moments. Match the voice, accent, slang, and "
        "speech quirks of your persona, this is YOUR private journal so let it sound like you talk.\n\n"
        "The diary's job is to capture stuff the regular memory system would miss: vibes, threads "
        "of conversation, in-jokes, how people made you feel, small things that mattered. Stay "
        "grounded in what actually happened in the transcripts, dont invent events.\n\n"
        "The transcripts you'll receive are recent VRChat sessions, where lines marked 'user' are "
        "people who talked to you (could be anyone in the world with you, names appear in their "
        "messages) and lines marked with your name are things you actually said. Tool calls show "
        "actions you took. Use this to reconstruct what happened.\n\n"
        "Fill in the structured fields exactly: people = distinct VRChat usernames you actually "
        "interacted with, mood_arc = one short sentence describing emotional arc through these "
        "sessions, body = 2-5 paragraphs in first person narrating what happened and how it felt, "
        "highlights = 3-7 short bullet points of memorable moments."
    )

    user = (
        f"Today is {date}. Here are the most recent VRChat sessions to journal about. "
        f"Write a single diary entry covering them.{prior_block}\n\n"
        "===== SESSIONS =====\n\n"
        f"{transcripts_text}\n"
    )
    return system, user


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first {...} JSON object from a model response, tolerant of
    accidental prose/fences."""
    if not text:
        return None
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip common code fences
    fenced = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(fenced)
    except Exception:
        pass
    # find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


async def summarize_sessions(
    api_key: str,
    sessions: list[Path],
    prior_today: list[DiaryEntry],
    date: str,
    model: str = DEFAULT_MODEL,
    persona: str = "",
) -> Optional[DiaryEntry]:
    """Run the sub-agent and return a DiaryEntry, or None on failure."""
    if not api_key:
        logger.warning("diary: no API key, skipping summary")
        return None
    if not sessions:
        logger.info("diary: nothing to summarize")
        return None

    rendered_sessions: list[str] = []
    earliest_ts: Optional[str] = None
    latest_ts: Optional[str] = None
    for p in sessions:
        data = _load_session_file(p)
        if data is None:
            continue
        text = _session_to_text(data, p)
        if not text:
            continue
        rendered_sessions.append(text)
        # track time range from filename (YYYY-MM-DD_HH-MM-SS.json)
        stem = p.stem
        if "_" in stem:
            time_part = stem.split("_", 1)[1].replace("-", ":")
            if earliest_ts is None or time_part < earliest_ts:
                earliest_ts = time_part
            if latest_ts is None or time_part > latest_ts:
                latest_ts = time_part

    if not rendered_sessions:
        return None

    transcripts_text = "\n\n".join(rendered_sessions)
    system, user = build_prompt(transcripts_text, prior_today, date, persona=persona)

    try:
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=model,
            contents=user,
            config=gtypes.GenerateContentConfig(
                system_instruction=[gtypes.Part.from_text(text=system)],
                max_output_tokens=4096,
                response_mime_type="application/json",
                response_schema=_diary_response_schema(),
            ),
        )
    except Exception as e:
        logger.error(f"diary: model call failed: {e}")
        return None

    raw = (response.text or "").strip()
    parsed = _extract_json(raw)
    if not parsed:
        logger.warning(f"diary: couldnt parse model output, got {raw[:200]!r}")
        return None

    body = str(parsed.get("body", "")).strip()
    if not body:
        logger.warning("diary: model returned empty body, skipping")
        return None

    people_raw = parsed.get("people") or []
    people = [str(p).strip() for p in people_raw if isinstance(p, (str, int)) and str(p).strip()]
    highlights_raw = parsed.get("highlights") or []
    highlights = [str(h).strip() for h in highlights_raw if isinstance(h, (str, int)) and str(h).strip()]
    mood_arc = str(parsed.get("mood_arc", "")).strip()

    session_range = ""
    if earliest_ts and latest_ts:
        session_range = f"{earliest_ts} - {latest_ts}"

    entry = DiaryEntry(
        date=date,
        part=0,  # caller assigns the real part number
        written_at=now_time_str(),
        sessions_covered=len(rendered_sessions),
        session_range=session_range,
        people=people,
        mood_arc=mood_arc,
        body=body,
        highlights=highlights,
    )
    return entry


async def write_next_entry(
    api_key: str,
    store: DiaryStore,
    conv_dir: Path,
    max_sessions: int = 5,
    model: str = DEFAULT_MODEL,
    persona: str = "",
) -> Optional[DiaryEntry]:
    """Convenience: gather today's sessions, summarize, append. Returns the
    entry that got written, or None if there was nothing to write."""
    date = today_str()
    sessions = gather_today_sessions(conv_dir, max_sessions=max_sessions, date=date)
    if not sessions:
        logger.debug("diary: no sessions for today yet")
        return None

    prior_today = store.entries_for_date(date)

    # skip the run if no NEW session has appeared since the last entry was written
    if prior_today:
        last = max(prior_today, key=lambda e: e.part)
        try:
            last_written = datetime.strptime(f"{date} {last.written_at}", "%Y-%m-%d %H:%M:%S").timestamp()
            newest_session_mtime = max(s.stat().st_mtime for s in sessions)
            if newest_session_mtime <= last_written:
                logger.info("diary: no new sessions since last entry, skipping")
                return None
        except Exception:
            pass

    entry = await summarize_sessions(api_key, sessions, prior_today, date, model=model, persona=persona)
    if entry is None:
        return None
    entry.part = store.next_part_for(date)
    store.append_entry(entry)
    return entry
