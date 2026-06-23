"""Quick manual check for the dynamic tool gate.

Loads the real config + tool registry, builds the gate, and prints what gets
sent for a few sample utterances plus the token savings vs sending everything.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# trigger @register_tool for every tool module the handler would load
from src.tools import (  # noqa: F401,E402
    avatar_scaling, emotions_tools, mapping, memory_tools, movement, music,
    personalities, soundboard, system, tracker, voice, vrchat_api, wanderer,
    yap_mode,
)
from src.tools import discord as _d  # noqa: F401,E402
from src.tools import music_gen as _mg  # noqa: F401,E402
from src.tools import social as _s  # noqa: F401,E402
from src.tools import time as _t  # noqa: F401,E402
from src.tools import web_search as _w  # noqa: F401,E402

from src.config import Config  # noqa: E402
from src.local_live.tools_adapter import collect_openai_tools  # noqa: E402
from src.local_live.tool_gate import ToolGate  # noqa: E402

import json  # noqa: E402

CORE = [
    "emotion", "stopAnimation",
    "saveMemory", "searchMemories", "recallMemories",
    "switchPersonality", "enableYapMode",
]


def _approx_tokens(tools):
    return len(json.dumps(tools)) // 4


def main():
    config = Config()
    all_tools = collect_openai_tools(config)
    gate = ToolGate(all_tools, core_names=CORE, max_dynamic=8)

    full_tokens = _approx_tokens(all_tools)
    base = gate.select("", [])
    print(f"total tools in catalog: {gate.total}")
    print(f"full payload ~{full_tokens} tokens")
    print(f"baseline (findTools + core): {len(base)} tools, ~{_approx_tokens(base)} tokens")
    print(f"core resolved: {[n for n in CORE if n in gate._by_name]}")
    missing = [n for n in CORE if n not in gate._by_name]
    if missing:
        print(f"  WARNING core names not found: {missing}")
    print()

    samples = [
        "play some music",
        "follow the player",
        "remember that my favorite color is blue",
        "search the web for the weather",
        "make yourself bigger",
        "go explore the world",
        "send my friend a message on discord",
    ]
    for s in samples:
        sel = gate.select(s, [])
        names = [t["function"]["name"] for t in sel]
        extra = [n for n in names if n not in CORE and n != "findTools"]
        print(f"{s!r}")
        print(f"  -> {len(sel)} tools (~{_approx_tokens(sel)} tok), matched: {extra}")

    print()
    print("findTools('navigate to a waypoint'):", gate.search("navigate to a waypoint"))
    print("findTools('change my avatar'):", gate.search("change my avatar"))
    print()
    avg = sum(_approx_tokens(gate.select(s, [])) for s in samples) / len(samples)
    print(f"avg sent ~{avg:.0f} tokens vs full ~{full_tokens} "
          f"({(1 - avg / full_tokens) * 100:.0f}% smaller)")


if __name__ == "__main__":
    main()
