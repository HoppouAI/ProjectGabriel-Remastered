"""LM Studio (OpenAI compatible) streaming client with tool calling + vision.

Why not the openai SDK? We already depend on httpx and the protocol is small
enough that hand-rolling SSE parsing keeps the dependency tree boring and
gives us tighter cancellation control.

Public surface:
    client = LMStudioClient(config)
    async for event in client.stream_turn(messages, tools, image_bytes=None):
        ...

Events yielded are tagged dicts:
    {"type": "text", "delta": str}
    {"type": "tool_call", "calls": [{"id","name","arguments_json"}]}
    {"type": "finish", "reason": str | None, "full_text": str}
    {"type": "error", "message": str}

The caller is responsible for: feeding deltas to TTS, dispatching tool calls,
appending tool messages, and looping again until finish.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


class LMStudioClient:
    def __init__(self, config):
        self.config = config
        self._base_url = config.local_llm_base_url
        self._model = config.local_llm_model
        self._api_key = config.local_llm_api_key
        self._temperature = config.local_llm_temperature
        self._top_p = config.local_llm_top_p
        self._max_tokens = config.local_llm_max_tokens
        self._timeout = config.local_llm_request_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def start(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, read=None),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )

    async def stop(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def encode_image(jpeg_bytes: bytes) -> dict:
        """Wrap JPEG bytes as the OpenAI vision content part."""
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }

    async def stream_turn(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[dict]:
        """Issue one chat completions request, yield streaming events."""
        if self._client is None:
            await self.start()
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        full_text_parts: list[str] = []
        # tool_call assembly: index -> dict with id/name/args fragments
        pending_tools: dict[int, dict] = {}
        finish_reason: Optional[str] = None

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    msg = f"LM Studio HTTP {resp.status_code}: {body.decode(errors='ignore')[:300]}"
                    yield {"type": "error", "message": msg}
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if "content" in delta and delta["content"]:
                        text = delta["content"]
                        full_text_parts.append(text)
                        yield {"type": "text", "delta": text}
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            slot = pending_tools.setdefault(
                                idx, {"id": "", "name": "", "arguments_json": ""}
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["arguments_json"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        except asyncio.CancelledError:
            raise
        except httpx.RequestError as e:
            yield {"type": "error", "message": f"LM Studio request failed: {e}"}
            return
        except Exception as e:
            yield {"type": "error", "message": f"LM Studio stream error: {e}"}
            return

        if pending_tools:
            yield {
                "type": "tool_call",
                "calls": [pending_tools[k] for k in sorted(pending_tools)],
            }

        yield {
            "type": "finish",
            "reason": finish_reason,
            "full_text": "".join(full_text_parts),
        }
