# Mood Plugin

A persistent 1-10 mood scale for the AI. The current mood gets injected into the system prompt every build, and the AI can change its own mood mid-conversation by calling the `setMood` tool.

- `1` = chill / no edge
- `10` = pissed off and uncensored

## Customizing the mood scale

Copy `moods.json.example` to `moods.json` in this folder and edit any of the levels. Each entry needs a `label` (short name) and a `vibe` (one or two sentence description of how the AI should act at that level).

```json
{
  "1": { "label": "zen",  "vibe": "pure zen mode, nothing fazes you" },
  "10": { "label": "NUKE", "vibe": "absolute scorched earth mode" }
}
```

Rules:
- Levels 1-10 only, anything outside that range is skipped with a warning.
- You don't have to provide all 10. Missing levels keep the built-in defaults.
- If `moods.json` is missing or invalid JSON, the defaults are used and a warning is logged.
- `moods.json` is gitignored so your edits stay local. `moods.json.example` is the tracked template.

## State

Current mood is persisted to `data/plugins/mood/state.json` and survives restarts.
