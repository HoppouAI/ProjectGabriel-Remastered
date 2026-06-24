"""Meta tool router for the Gemini Live session.

Gemini Live sets its tool list once at connect time over a persistent
websocket, so we cant swap declarations per turn like the local backend
does. Instead we hand the model two meta tools, searchTools and
executeTool, plus a small hot core, and dispatch everything else by name
through the normal handler. cuts the declaration payload from 100ish full
schemas down to a handful.
"""

import logging
import re

from google.genai import types

from src.emotions import generate_emotion_function_declarations
from src.tools._base import get_registered_tools

logger = logging.getLogger(__name__)

# always declared directly, called too often to eat a search round trip
CORE_TOOL_NAMES = {
    "emotion",
    "stopAnimation",
    "saveMemory",
    "searchMemories",
    "recallMemories",
    "switchPersonality",
    "enableYapMode",
}

META_TOOL_NAMES = {"searchTools", "executeTool"}

_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

_STOP = {
    "the", "a", "an", "to", "of", "for", "and", "or", "in", "on", "with",
    "my", "me", "i", "you", "your", "it", "that", "this", "is", "are", "do",
    "can", "please", "want", "need", "let", "get", "go", "use", "call",
}

# voice phrasings the model might use mapped to words that show up in tool
# names/descriptions. keep it small, just the common stuff
_SYNONYMS = {
    "play": ("music", "song", "sound", "soundboard"),
    "song": ("music", "suno", "generate"),
    "music": ("song", "suno", "play"),
    "sound": ("soundboard", "sfx", "play"),
    "follow": ("track", "person", "tracker"),
    "track": ("follow", "person"),
    "look": ("face", "tracker", "wander"),
    "face": ("tracker", "look"),
    "wander": ("explore", "move", "walk"),
    "explore": ("wander", "move"),
    "walk": ("move", "wander"),
    "avatar": ("switch", "change"),
    "scale": ("size", "grow", "shrink", "height"),
    "size": ("scale",),
    "grow": ("scale",),
    "shrink": ("scale",),
    "volume": ("loud", "quiet"),
    "web": ("search", "google", "lookup", "internet"),
    "search": ("web", "lookup", "find"),
    "friend": ("social", "message", "dm"),
    "message": ("social", "dm", "friend"),
    "dm": ("message", "social"),
    "voice": ("tts", "accent"),
    "accent": ("voice", "tts"),
    "map": ("mapping", "waypoint", "navigate"),
    "navigate": ("map", "waypoint", "path"),
    "where": ("map", "location"),
}


def _tokenize(text):
    return [t.lower() for t in _WORD_RE.findall(text or "") if t.lower() not in _STOP]


def _expand(tokens):
    out = set(tokens)
    for t in tokens:
        out.update(_SYNONYMS.get(t, ()))
    return out


def _schema_to_dict(schema):
    if schema is None:
        return {"type": "object", "properties": {}}
    if isinstance(schema, dict):
        return schema
    for kwargs in ({"exclude_none": True, "mode": "json"}, {"exclude_none": True}):
        try:
            return schema.model_dump(**kwargs)
        except Exception:
            continue
    return {}


def _short_desc(desc, limit=140):
    if not desc:
        return ""
    # drop the invocation condition tail we tack onto every tool
    head = desc.split("\n**Invocation Condition:**")[0].strip()
    head = head.split("\n")[0].strip()
    if len(head) > limit:
        head = head[: limit - 1].rstrip() + "…"
    return head


def _collect_flat(config):
    """Same flat function declaration list config_builder would send, minus
    the Tool wrapping. mirrors get_tool_declarations so the index matches
    what the model could actually call."""
    decls = []
    for cls in get_registered_tools():
        inst = cls.__new__(cls)
        inst.handler = None
        try:
            decls.extend(inst.declarations(config=config) or [])
        except Exception:
            continue
    if config:
        for d in generate_emotion_function_declarations(config):
            decls.append(types.FunctionDeclaration(
                name=d["name"], description=d["description"], parameters=d["parameters"],
            ))
    if config and hasattr(config, "is_tool_enabled"):
        decls = [d for d in decls if config.is_tool_enabled(d.name)]
    return decls


def _meta_function_declarations():
    return [
        types.FunctionDeclaration(
            name="searchTools",
            description=(
                "Find a tool you can run. Pass a short query describing what you want "
                "to do (eg 'play a song', 'follow that person', 'change my avatar'). "
                "Returns matching tool names with their argument schemas. Use this "
                "before executeTool whenever you need something that isnt already a "
                "direct tool.\n**Invocation Condition:** Call when you want to act and "
                "dont see a direct tool for it."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "what you want to do, a few words"},
                },
                "required": ["query"],
            },
        ),
        types.FunctionDeclaration(
            name="executeTool",
            description=(
                "Run a tool by name with JSON arguments. Get the exact name and "
                "argument shape from searchTools first. Never mention tool names or "
                "this mechanism out loud, just do the action and talk naturally."
                "\n**Invocation Condition:** Call to actually perform an action after "
                "finding the tool with searchTools."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "tool": {"type": "STRING", "description": "exact tool name from searchTools"},
                    "args_json": {
                        "type": "STRING",
                        "description": "arguments as a JSON object string, eg {\"track\":\"lofi\"}. use {} for no args",
                    },
                },
                "required": ["tool"],
            },
        ),
    ]


def build_meta_declarations(config):
    """Tool list for the Live config when dynamic tools are on: the two meta
    tools + the hot core, wrapped the same way get_tool_declarations does."""
    flat = _collect_flat(config)
    core = [d for d in flat if d.name in CORE_TOOL_NAMES]
    fn_decls = _meta_function_declarations() + core
    tools = []
    if config and config.google_search_enabled:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    tools.append(types.Tool(function_declarations=fn_decls))
    return tools


def build_meta_prompt(config):
    """Catalog + usage block appended to the system prompt so the model knows
    what exists without us shipping every full schema."""
    flat = _collect_flat(config)
    rows = sorted(
        (d.name, _short_desc(d.description))
        for d in flat
        if d.name not in CORE_TOOL_NAMES and d.name not in META_TOOL_NAMES
    )
    if not rows:
        return ""
    lines = [
        "## Toolset (on demand)",
        "Most of your abilities are not declared directly to save space. To use one:",
        "1. call searchTools with a short query for what you want to do",
        "2. take the exact tool name and argument shape from the result",
        "3. call executeTool with that name and args_json as a JSON object string ({} if no args)",
        "Core tools (emotions, memory, personality, yap mode) are always declared, never route those through executeTool.",
        "Never say tool names or mention this system out loud, just act and talk normally.",
        "",
        "### Available tools",
    ]
    lines += [f"- {name}: {desc}" if desc else f"- {name}" for name, desc in rows]
    return "\n".join(lines)


class MetaToolRouter:
    """Owns the searchable index and answers searchTools queries. executeTool
    dispatch itself runs through ToolHandler.handle_by_name."""

    def __init__(self, config):
        self.config = config
        self._index = None
        self._known = None

    def _ensure(self):
        if self._index is not None:
            return
        flat = _collect_flat(self.config)
        self._known = {d.name for d in flat}
        idx = {}
        for d in flat:
            if d.name in META_TOOL_NAMES or d.name in CORE_TOOL_NAMES:
                continue
            idx[d.name] = {
                "name": d.name,
                "description": d.description or "",
                "parameters": _schema_to_dict(d.parameters),
                "name_tokens": set(_tokenize(d.name)),
                "desc_tokens": set(_tokenize(d.description or "")),
            }
        self._index = idx
        logger.info(f"meta tool router: indexed {len(idx)} tools behind searchTools/executeTool")

    def is_known(self, name):
        self._ensure()
        return name in self._known and name not in META_TOOL_NAMES

    def search(self, query, top=6):
        self._ensure()
        terms = _expand(_tokenize(query))
        scored = []
        for entry in self._index.values():
            score = 0
            for t in terms:
                if t in entry["name_tokens"]:
                    score += 3
                elif t in entry["desc_tokens"]:
                    score += 1
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"name": e["name"], "description": e["description"], "parameters": e["parameters"]}
            for _s, e in scored[:top]
        ]
