# Mood Plugin

A persistent mood system for the AI with two dimensions:

- **emotion**: what the AI is feeling (`happy`, `sad`, `scared`, `angry`, `amused`, `lonely`, ...)
- **intensity**: how strongly it feels it, on a 1-10 scale

Combined, these get injected into the system prompt at every Gemini session start, and the AI can change its own mood mid-conversation by calling the `setMood` tool. Mood persists across restarts via `data/plugins/mood/state.json`.

## Default emotions

`neutral, happy, excited, content, amused, annoyed, frustrated, angry, furious, sad, lonely, scared, anxious, confused, curious, proud, embarrassed, disgusted, surprised, playful, uncensored`

## Customizing

Both files live in this folder and are gitignored, only the `.example` files are tracked.

### Override the intensity scale

Copy `moods.json.example` to `moods.json` and edit any of the levels. Each entry needs a `label` (short name) and a `vibe` (one or two sentence description).

```json
{
  "1":  { "label": "zen",  "vibe": "pure zen mode, nothing fazes you" },
  "10": { "label": "NUKE", "vibe": "absolute scorched earth mode" }
}
```

### Add or override emotions

Copy `emotions.json.example` to `emotions.json` and edit it. Each entry is an emotion name (lowercase) mapping to a `vibe` describing how it shows up in behavior. You can override built-in emotions or add brand new ones.

```json
{
  "smug":      { "vibe": "insufferably pleased with yourself, gloats lightly, drops 'I told you so' a lot" },
  "wholesome": { "vibe": "soft hearted, encouraging, gently supportive of everyone" }
}
```

## Rules

- Levels 1-10 only, anything outside that range is skipped with a warning
- You don't have to provide all 10 levels or all emotions, missing entries keep the built-in defaults
- If either file is missing or invalid JSON, defaults are used and a warning is logged
- Emotion names are normalized to lowercase
- Keys starting with `_` (like `_comment`) are silently ignored
