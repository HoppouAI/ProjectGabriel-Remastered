"""Background JSON calls to Gemini that walk the key list on rate limits without touching the live session's key."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from google import genai
from google.genai import types

from src.quota import is_rate_limit

logger = logging.getLogger(__name__)


def parse_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


async def generate_json(config, model: str, system: str, prompt: str, timeout: float = 120.0):
    keys = list(getattr(config, "_keys", None) or [config.api_key])
    start = getattr(config, "_key_index", 0)
    last = None
    for i in range(len(keys)):
        key = keys[(start + i) % len(keys)]
        try:
            client = genai.Client(api_key=key)
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        temperature=0.3,
                        max_output_tokens=8192,
                    ),
                ),
                timeout=timeout,
            )
            return parse_json(resp.text)
        except Exception as e:
            last = e
            if is_rate_limit(e) and i < len(keys) - 1:
                logger.info(f"mind: key {i + 1}/{len(keys)} rate limited, trying the next one")
                continue
            raise
    raise last
