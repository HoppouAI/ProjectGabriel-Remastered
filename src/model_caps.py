"""Per-model capability profiles for the Gemini Live API. One table, everything reads it."""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapabilities:
    name: str
    family: str
    generation: float
    known: bool = True
    thinking: str = "none"            # budget (2.5) | level (3.x) | none (built in)
    thinking_levels: tuple = ()       # accepted values when thinking == "level"
    text_input: str = "realtime"      # realtime (send_realtime_input) | client_content
    history_config: bool = True       # needs initial_history_in_client_content
    alpha_features: bool = False      # affective dialog + proactivity params
    google_search: bool = False       # default for built-in search grounding

    vision_token_cap: bool = True     # shrink frames + slow the loop to save tokens
    async_tools: str = "optional"     # none | optional | required (required = blocking errors)
    tool_scheduling: bool = True      # SILENT/WHEN_IDLE/INTERRUPT on tool responses
    turn_complete_is_idle: bool = True  # false when the model keeps reasoning after a turn
    proactive_audio_forced: bool = False
    vision_max_size: int = 768
    vision_quality: int = 60
    vision_min_interval: float = 2.0


# key = model name prefix, longest match wins so extended-thinking beats plain 3.8
# NOTE: grounding is off for everything. it has its own quota separate from the model
# quota, free tier keys effectively get none of it, and when it fails it 1011s with the
# same message as a dead key so rotation burns the whole pool finding nothing. the app's
# own webSearch tool covers search for free. set google_search: true in config.yml to opt in.
_KNOWN_MODELS = {
    "gemini-2.5-flash-native-audio-preview": dict(
        family="2.5",
        generation=2.5,
        thinking="budget",
        text_input="client_content",
        history_config=False,
        alpha_features=True,
        google_search=False,
        vision_token_cap=False,
        async_tools="none",
        tool_scheduling=False,
    ),
    "gemini-3.1-flash-live": dict(
        family="3.1",
        generation=3.1,
        thinking="level",
        thinking_levels=("minimal", "low", "medium", "high"),
        google_search=False,
        async_tools="none",
        tool_scheduling=False,
    ),
    "gemini-3.8-live-extended-thinking": dict(
        family="3.8-extended-thinking",
        generation=3.8,
        thinking="level",
        thinking_levels=("low", "medium", "high"),
        # grounding has its own quota free keys effectively dont get, a 1011 here looks
        # exactly like a dead key and burns the whole rotation pool
        google_search=False,
        async_tools="required",
        tool_scheduling=False,
        turn_complete_is_idle=False,
        proactive_audio_forced=True,
    ),
    "gemini-3.8-live": dict(
        family="3.8",
        generation=3.8,
        # interleaved reasoning, always on, not configurable.
        thinking="none",
        google_search=False,
        async_tools="optional",
        proactive_audio_forced=True,
    ),
}


# canonical ordering, used to pick the nearest allowed level when one is rejected
_LEVEL_ORDER = ("minimal", "low", "medium", "high")

# per-tool function call types an operator can set in tools.yml
TOOL_TYPES = ("auto", "blocking", "non_blocking")
TOOL_TYPE_DEFAULT = "auto"

# how the model should treat a result that arrives while it's busy
TOOL_SCHEDULING = ("when_idle", "interrupt", "silent")
TOOL_SCHEDULING_DEFAULT = "when_idle"


@dataclass
class CompatNote:
    """A setting we changed or dropped because the model would reject it."""

    setting: str
    reason: str

    def __str__(self):
        return f"{self.setting}: {self.reason}"


def resolve_model(model_name: str) -> ModelCapabilities:
    """Look up a model by name, falling back to guessing from its version."""
    name = (model_name or "").strip()
    lowered = name.lower()

    for key in sorted(_KNOWN_MODELS, key=len, reverse=True):
        if lowered.startswith(key):
            return ModelCapabilities(name=name, **_KNOWN_MODELS[key])

    # unknown model. guess from the version number so a future release still runs.
    match = re.search(r"gemini-(\d+)\.(\d+)", lowered)
    if match:
        generation = float(f"{match.group(1)}.{match.group(2)}")
        if generation >= 3.8:
            profile = dict(_KNOWN_MODELS["gemini-3.8-live"])
        elif generation >= 3.0:
            profile = dict(_KNOWN_MODELS["gemini-3.1-flash-live"])
        else:
            profile = dict(_KNOWN_MODELS["gemini-2.5-flash-native-audio-preview"])
        logger.warning(
            f"unknown Live model '{name}', assuming {profile['family']} behaviour "
            f"based on the version number. things may need tweaking."
        )
        return ModelCapabilities(name=name, known=False, **profile)

    logger.warning(
        f"unrecognised model name '{name}', assuming newest Live behaviour. "
        f"if it misbehaves, add a profile for it in src/model_caps.py"
    )
    return ModelCapabilities(name=name, known=False, **_KNOWN_MODELS["gemini-3.8-live"])


def _nearest_level(level: str, allowed: tuple):
    if not allowed:
        return None
    want = level.lower().strip()
    if want not in _LEVEL_ORDER:
        return allowed[0]
    target = _LEVEL_ORDER.index(want)
    return min(allowed, key=lambda a: abs(_LEVEL_ORDER.index(a) - target))


def resolve_thinking(
    caps: ModelCapabilities,
    *,
    enabled: bool = True,
    budget=None,
    level=None,
    include_thoughts: bool = False,
):
    """Turn whatever the user configured into what this model accepts, never raises."""
    notes: list[CompatNote] = []

    if not enabled:
        return None, notes

    requested = []
    if budget is not None:
        requested.append(f"token budget {budget}")
    if level is not None:
        requested.append(f"level '{level}'")

    # model reasons on its own, any thinking param is rejected outright
    if caps.thinking == "none":
        if requested or include_thoughts:
            what = ", ".join(requested) if requested else "thought summaries"
            notes.append(CompatNote(
                "thinking",
                f"{caps.name} does not accept a thinking config (reasoning is built in "
                f"and always on), ignoring {what}",
            ))
        return None, notes

    kwargs = {}

    if caps.thinking == "budget":
        if level is not None:
            notes.append(CompatNote(
                "thinking level",
                f"{caps.name} is a 2.5 model and takes a token budget instead of levels, "
                f"ignoring level '{level}'",
            ))
        if budget is not None:
            kwargs["thinking_budget"] = budget

    else:  # thinking == "level"
        if budget is not None:
            notes.append(CompatNote(
                "thinking budget",
                f"{caps.name} takes thinking levels instead of a token budget, "
                f"ignoring budget {budget}",
            ))
        if level is not None:
            if level.lower().strip() in caps.thinking_levels:
                kwargs["thinking_level"] = level.lower().strip()
            else:
                snapped = _nearest_level(level, caps.thinking_levels)
                notes.append(CompatNote(
                    "thinking level",
                    f"{caps.name} does not support '{level}', using '{snapped}' instead "
                    f"(allowed: {', '.join(caps.thinking_levels)})",
                ))
                if snapped:
                    kwargs["thinking_level"] = snapped

    if include_thoughts:
        kwargs["include_thoughts"] = True

    return (kwargs or None), notes


def resolve_tool_behavior(caps: ModelCapabilities, requested: str = TOOL_TYPE_DEFAULT):
    """Map a tools.yml type onto a behavior. None means leave the model default alone."""
    want = (requested or TOOL_TYPE_DEFAULT).strip().lower().replace("-", "_")

    if want not in TOOL_TYPES:
        return None, f"'{requested}' is not a valid tool type"

    # no async support at all, the model only does blocking calls
    if caps.async_tools == "none":
        if want == "non_blocking":
            return None, "blocking_only"
        return None, None

    # async is mandatory here, blocking is a hard error
    if caps.async_tools == "required":
        if want == "blocking":
            return "NON_BLOCKING", "forced_async"
        return "NON_BLOCKING", None

    # async available but optional, honor whatever was asked for
    if want == "blocking":
        return "BLOCKING", None
    if want == "non_blocking":
        return "NON_BLOCKING", None
    return None, None


def resolve_tool_scheduling(
    caps: ModelCapabilities,
    requested=None,
    behavior: str = None,
):
    """Map a tools.yml scheduling onto a FunctionResponseScheduling name, dropping it when unusable."""
    effective_async = behavior == "NON_BLOCKING" or (
        behavior is None and caps.async_tools in ("optional", "required")
    )

    if requested is None or str(requested).strip() == "":
        # auto-attach only when useful, else the result never gets mentioned
        if effective_async and caps.tool_scheduling:
            return TOOL_SCHEDULING_DEFAULT.upper(), None
        return None, None

    want = str(requested).strip().lower().replace("-", "_")
    if want not in TOOL_SCHEDULING:
        return None, f"'{requested}' is not a valid scheduling value"

    if not caps.tool_scheduling:
        return None, "no_scheduling"
    if not effective_async:
        return None, "not_async"
    return want.upper(), None


def check_alpha_features(
    caps: ModelCapabilities,
    *,
    affective_dialog=None,
    proactivity=None,
):
    """Report alpha-only settings the model will reject. Returns CompatNote list."""
    notes: list[CompatNote] = []
    if affective_dialog is None and proactivity is None:
        return notes
    if caps.alpha_features:
        return notes

    offenders = []
    if affective_dialog is not None:
        offenders.append("enable_affective_dialog")
    if proactivity is not None:
        offenders.append("proactivity")

    if caps.proactive_audio_forced:
        reason = f"{caps.name} has proactive audio permanently on and no affective dialog, ignoring them"
    else:
        reason = f"{caps.name} is a 3.x model and does not support these, ignoring them"

    notes.append(CompatNote(", ".join(offenders), reason))
    return notes


def interaction_status(server_content):
    """IDLE, IN_PROGRESS or REQUIRES_ACTION, or None when the server didnt say."""
    status = getattr(server_content, "interaction_status", None)
    if status is None:
        return None
    name = str(getattr(status, "value", status)).upper()
    if name.endswith("UNSPECIFIED"):
        return None
    for known in ("IN_PROGRESS", "REQUIRES_ACTION", "IDLE"):
        if name.endswith(known):
            return known
    return None


def interaction_is_idle(server_content) -> bool:
    """True when the model reported interaction_status=IDLE, absent counts as not idle."""
    return interaction_status(server_content) == "IDLE"


def describe(caps: ModelCapabilities) -> str:
    """Short human readable summary for the startup log."""
    bits = [f"family {caps.family}"]
    if caps.thinking == "budget":
        bits.append("thinking budget")
    elif caps.thinking == "level":
        bits.append(f"thinking levels ({'/'.join(caps.thinking_levels)})")
    else:
        bits.append("built in reasoning")
    if caps.async_tools != "none":
        bits.append(f"async tools ({caps.async_tools})")
    if caps.turn_complete_is_idle is False:
        bits.append("needs interaction_status")
    return ", ".join(bits)
