# Diary plugin

Long term life diary for Gabriel. A background sub-agent reads recent VRChat
session transcripts every couple hours and writes a first person diary entry
to a custom `.diary` file. The AI gets tools to read its own diary back when
it needs context the structured memory system would not capture.

## What it does

- Background scheduler runs every 2 hours (configurable) inside the host event loop.
- Each tick gathers the **last N session transcripts from today** (default 5) from `data/conversations/`.
- Passes them to `gemini-3.1-flash-lite-preview` (configurable) along with any
  earlier diary entries from today, so the new entry builds forward instead of
  repeating itself.
- Writes a structured entry to `data/plugins/diary/gabriel.diary`.
- A diary "day" can have multiple "parts" (one per scheduler tick that wrote
  something new), so a busy day looks like:

```
=== 2026-05-08 part 1 (written May 8, 2026 at 2:30:11 PM) ===
...

=== 2026-05-08 part 2 (written May 8, 2026 at 4:30:42 PM) ===
...
```

## Tools the AI can call

| name | purpose |
|---|---|
| `readDiary` | read entries by date or get the most recent N |
| `searchDiary` | substring search across all entries |
| `listDiaryDates` | list every date that has at least one entry |
| `updateDiaryNow` | force the scheduler to run a tick immediately |

## Config (optional, all fields default)

```yaml
plugins:
  diary:
    enabled: true
    interval_hours: 2          # how often the background scheduler runs
    max_sessions: 5            # how many recent today-sessions to summarize per tick
    model: "gemini-3.1-flash-lite-preview"
    initial_delay_seconds: 300 # warmup delay after startup before first tick
    filename: "gabriel.diary"  # name of the diary file under data/plugins/diary/
    conversation_dir: "data/conversations"  # where session transcripts live
```

## File format

Plain text, easy to open in any editor. Each entry is bracketed by
`=== DATE part N (written FRIENDLY_TIMESTAMP) ===` and `=== END ===` markers,
with a small metadata block, the body paragraphs, and an optional bullet point
highlights list. Timestamps use US 12-hour format (e.g. `May 8, 2026 at 2:30:11 PM`)
to match what the AI's time tool returns, so the model doesnt get confused
flipping between 24h and 12h. Parser is lenient: hand edits and missing fields are fine.

## Notes

- Requires `privacy.save_conversations: true` in the main `config.yml`,
  otherwise no transcripts are written and the diary stays empty.
- The diary is meant to capture **vibes and threads** that the structured
  memory tools miss. Names, ongoing jokes, how people made you feel.
- The plugin never edits or deletes past entries, only appends new ones.
- API key comes from the same `Config.api_key` rotation used by the main session.
