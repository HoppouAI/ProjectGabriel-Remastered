"""Adapt the Gemini-shaped tool registry to OpenAI-compatible tool schema.

The ToolHandler in src.tools knows about google.genai types. LM Studio (and
every other OpenAI-compatible server) expects the chat-completions tool spec:

    {
      "type": "function",
      "function": {
        "name": "...",
        "description": "...",
        "parameters": { JSON schema }
      }
    }

We walk every registered tool, ask it for its FunctionDeclaration list, and
convert each one. The parameters block from google-genai is already a JSON
schema dict (or convertible to one), so the heavy lifting is just type-name
casing.
"""

from __future__ import annotations

import logging
from typing import Any

from google.genai import types

from src.emotions import generate_emotion_function_declarations
from src.tools._base import get_registered_tools

logger = logging.getLogger(__name__)


def _normalize_schema(schema: Any) -> dict:
    """Recursively turn a google.genai Schema/dict into a plain JSON schema
    with lowercased type names (LM Studio's strict mode rejects uppercase)."""
    if schema is None:
        return {}
    # google.genai Schema objects expose to_json_dict() via model_dump
    if hasattr(schema, "model_dump"):
        try:
            schema = schema.model_dump(exclude_none=True)
        except Exception:
            pass
    if not isinstance(schema, dict):
        return {}

    out: dict = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            out[k] = v.lower()
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: _normalize_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _normalize_schema(v)
        elif k == "any_of":
            out["anyOf"] = [_normalize_schema(s) for s in (v or [])]
        elif k == "anyOf" and isinstance(v, list):
            out[k] = [_normalize_schema(s) for s in v]
        else:
            out[k] = v
    # OpenAI tools want object schemas to have properties+required even if empty
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


def collect_openai_tools(config) -> list[dict]:
    """Walk the global tool registry + emotion declarations, filter by
    config.is_tool_enabled, return an OpenAI-compatible tools list."""
    decls: list[types.FunctionDeclaration] = []
    for cls in get_registered_tools():
        try:
            probe = cls.__new__(cls)
            probe.handler = None
            decls.extend(probe.declarations(config=config) or [])
        except Exception as e:
            logger.debug(f"tool {cls.__name__} declarations() failed: {e}")

    # emotion functions are produced separately like in gemini mode
    if config:
        try:
            for d in generate_emotion_function_declarations(config):
                decls.append(types.FunctionDeclaration(
                    name=d["name"],
                    description=d["description"],
                    parameters=d["parameters"],
                ))
        except Exception as e:
            logger.warning(f"emotion declarations failed: {e}")

    enabled = []
    for d in decls:
        name = getattr(d, "name", None)
        if not name:
            continue
        if config and hasattr(config, "is_tool_enabled") and not config.is_tool_enabled(name):
            continue
        enabled.append(d)

    tools = []
    seen = set()
    for d in enabled:
        if d.name in seen:
            continue
        seen.add(d.name)
        params = _normalize_schema(getattr(d, "parameters", None)) or {
            "type": "object", "properties": {}
        }
        tools.append({
            "type": "function",
            "function": {
                "name": d.name,
                "description": getattr(d, "description", "") or "",
                "parameters": params,
            },
        })
    return tools
