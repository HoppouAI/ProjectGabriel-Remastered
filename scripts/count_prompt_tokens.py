"""Count tokens of the full system prompt + tool declarations as sent to Gemini Live.

Builds the same system_instruction and tools list that config_builder._build_config
hands to client.aio.live.connect, then runs client.models.count_tokens on each piece
plus the combined payload. Prints a breakdown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from google import genai
from google.genai import types

from src.config import Config
from src.personalities import PersonalityManager
from src.tools import get_tool_declarations


def main():
    cfg = Config()
    pm = None
    try:
        pm = PersonalityManager(cfg)
    except Exception as e:
        print(f"(personality manager skipped: {e})")

    sysinst_text = cfg.build_system_instruction(pm)
    tool_decls = get_tool_declarations(cfg)

    # Flatten tool declarations into a single dict payload we can count.
    tools_payload = []
    flat_decl_count = 0
    for tool in tool_decls:
        fdecls = getattr(tool, "function_declarations", None) or []
        for fd in fdecls:
            flat_decl_count += 1
            tools_payload.append({
                "name": fd.name,
                "description": fd.description,
                "parameters": fd.parameters.model_dump() if hasattr(fd, "parameters") and fd.parameters else None,
            })
    tools_json = json.dumps(tools_payload, indent=2)

    client = genai.Client(api_key=cfg.api_key)
    # Use a stable counting model. Live models can't be used with count_tokens directly;
    # the tokenizer for gemini-2.5-flash matches the live family closely enough for sizing.
    count_model = "gemini-2.5-flash"

    def count(label, text):
        try:
            r = client.models.count_tokens(model=count_model, contents=text)
            tokens = r.total_tokens
        except Exception as e:
            tokens = f"ERR {e}"
        chars = len(text)
        print(f"{label:<32} tokens={tokens:<8} chars={chars}")
        return tokens if isinstance(tokens, int) else 0

    print(f"Live model in use:   {cfg.model}")
    print(f"Counting against:    {count_model}")
    print(f"Tool declarations:   {flat_decl_count} functions across {len(tool_decls)} Tool groups")
    print("-" * 70)

    sys_tokens = count("system_instruction", sysinst_text)
    tools_tokens = count("tools (json dump)", tools_json)
    combined = sysinst_text + "\n\n--- TOOLS ---\n" + tools_json
    total_tokens = count("combined payload", combined)

    print("-" * 70)
    print(f"system_instruction:  {sys_tokens} tokens, {len(sysinst_text)} chars")
    print(f"tools payload:       {tools_tokens} tokens, {len(tools_json)} chars")
    print(f"combined:            {total_tokens} tokens")
    print()
    print("Per-tool breakdown:")
    for entry in sorted(tools_payload, key=lambda e: -len(json.dumps(e))):
        j = json.dumps(entry)
        try:
            t = client.models.count_tokens(model=count_model, contents=j).total_tokens
        except Exception as e:
            t = f"ERR {e}"
        print(f"  {entry['name']:<32} tokens={t:<6} chars={len(j)}")


if __name__ == "__main__":
    main()
