"""Repro for the multi turn tool calling stall against the live llama-server.

Mirrors LocalLiveSession._run_turn's loop: select tools, stream, collect tool
calls, dispatch (findTools via the gate, everything else mocked), loop. Prints
per-iteration what came back (thought chars, visible text, tool calls, finish
reason) so we can see exactly where it stops producing a spoken reply.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.local_live.llm import LMStudioClient  # noqa: E402
from src.local_live.tools_adapter import collect_openai_tools  # noqa: E402
from src.local_live.tool_gate import ToolGate  # noqa: E402

CORE = [
    "emotion", "stopAnimation",
    "saveMemory", "searchMemories", "recallMemories",
    "switchPersonality", "enableYapMode",
]


def mock_tool(name, args):
    # plausible-ish results so the model has something to talk about
    if name in ("searchSoundboard",):
        return {"result": "ok", "results": ["airhorn", "vine boom", "bruh"]}
    if name in ("playSoundboard", "playRandomSoundboard"):
        return {"result": "ok", "played": "vine boom"}
    return {"result": "ok"}


async def run_turn(client, gate, system_text, history, user_text, max_iter=6, stall_iter=None):
    activated: list[str] = []
    messages = [{"role": "system", "content": system_text}] + history
    messages.append({"role": "user", "content": user_text})
    history.append({"role": "user", "content": user_text})

    full_text = ""
    forced_answer = False
    print(f"\n=== USER: {user_text!r} (stall_iter={stall_iter}) ===")
    for it in range(max_iter):
        if forced_answer:
            tools = None
        else:
            tools = gate.select(user_text, activated)
        tool_names = [t["function"]["name"] for t in (tools or [])]
        thought_chars = 0
        iter_text = ""
        calls = []
        finish = None

        if it == stall_iter and not forced_answer:
            # simulate the model going quiet after a tool: only reasons, no text
            thought_chars = 120
            finish = "stop"
            print(f"  iter {it}: [INJECTED SILENT RESPONSE] thought-only, no text/tools")
        else:
            async for ev in client.stream_turn(messages, tools=tools):
                t = ev.get("type")
                if t == "thought":
                    thought_chars += len(ev.get("delta", ""))
                elif t == "text":
                    iter_text += ev["delta"]
                elif t == "tool_call":
                    calls = ev["calls"]
                elif t == "finish":
                    finish = ev.get("reason")
                elif t == "error":
                    print(f"  [ERROR] {ev['message']}")
                    return full_text
            full_text += iter_text
            call_desc = [f"{c['name']}({c['arguments_json']})" for c in calls]
            print(f"  iter {it}: sent {len(tool_names)} tools | thought={thought_chars}ch "
                  f"text={iter_text.strip()[:80]!r} finish={finish} calls={call_desc}")

        if calls:
            assistant_msg = {
                "role": "assistant",
                "content": iter_text or None,
                "tool_calls": [
                    {"id": c.get("id") or f"call_{i}", "type": "function",
                     "function": {"name": c["name"], "arguments": c.get("arguments_json") or "{}"}}
                    for i, c in enumerate(calls)
                ],
            }
            messages.append(assistant_msg)
            history.append(assistant_msg)

            for c in calls:
                name = c.get("name") or ""
                try:
                    args = json.loads(c.get("arguments_json") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "findTools":
                    q = args.get("query", "")
                    found = gate.search(q, top=8)
                    for n in found:
                        if n not in activated:
                            activated.append(n)
                    result = {"activated": found, "note": "now available, call them on your next step"}
                    print(f"    findTools({q!r}) -> {found}")
                else:
                    result = mock_tool(name, args)
                    print(f"    {name}({args}) -> {result}")
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": c.get("id") or f"call_{name}",
                    "name": name,
                    "content": json.dumps(result),
                }
                messages.append(tool_msg)
                history.append(tool_msg)
            continue

        # no tool calls. mirror the session recovery: if nothing was spoken,
        # force one more pass with no tools so the turn isn't silent.
        if not full_text.strip() and not forced_answer:
            forced_answer = True
            print("    [recovery] empty response, forcing a spoken reply with no tools")
            continue
        break

    ai_text = full_text.strip()
    print(f"  >>> FINAL ai_text: {ai_text[:160]!r}")
    if not ai_text:
        print("  !!! SILENT TURN (no spoken reply) -- this is the bug")
    else:
        print("  OK spoken reply produced")
    history.append({"role": "assistant", "content": ai_text})
    return ai_text


async def main():
    config = Config()
    system_text = config.build_system_instruction(None)
    from src.local_live.session import _DYNAMIC_TOOLS_NOTE, _CONCISE_REASONING_NOTE
    if config.local_llm_concise_reasoning:
        system_text += _CONCISE_REASONING_NOTE
    if config.local_llm_dynamic_tools:
        system_text += _DYNAMIC_TOOLS_NOTE

    all_tools = collect_openai_tools(config)
    gate = ToolGate(all_tools, core_names=CORE, max_dynamic=config.local_llm_dynamic_tools_max)
    print(f"dynamic_tools={config.local_llm_dynamic_tools} system_text={len(system_text)}ch "
          f"catalog={gate.total}")

    client = LMStudioClient(config)
    await client.start()
    try:
        history: list[dict] = []
        # normal multi turn tool calls
        await run_turn(client, gate, system_text, history, "Can you play uh some sound effect?")
        await run_turn(client, gate, system_text, history, "Play a random meme sound.")
        # inject the silent failure: model uses a tool then returns no text on
        # the next step. with the recovery this should still end up speaking.
        await run_turn(client, gate, system_text, history, "Check your system specs.", stall_iter=1)
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
