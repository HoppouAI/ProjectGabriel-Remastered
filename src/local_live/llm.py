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
    {"type": "thought", "delta": str}
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
import re
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


class _ThinkTagFilter:
    """Strip <think>...</think> spans out of streamed text.

    Emits two streams (visible_text, thought_text) from chunks that may split
    a tag across boundaries. Handles the common reasoning-model conventions:
    <think>...</think> (DeepSeek R1, Qwen3, GPT-OSS), case-insensitive.
    """

    _OPEN = re.compile(r"<think>", re.IGNORECASE)
    _CLOSE = re.compile(r"</think>", re.IGNORECASE)

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str) -> tuple[str, str]:
        self._buf += chunk
        out_visible = []
        out_thought = []
        while self._buf:
            if self._in_think:
                m = self._CLOSE.search(self._buf)
                if not m:
                    # keep last 8 chars in case </think> straddles boundary
                    if len(self._buf) > 8:
                        out_thought.append(self._buf[:-8])
                        self._buf = self._buf[-8:]
                    break
                out_thought.append(self._buf[:m.start()])
                self._buf = self._buf[m.end():]
                self._in_think = False
            else:
                m = self._OPEN.search(self._buf)
                if not m:
                    if len(self._buf) > 7:
                        out_visible.append(self._buf[:-7])
                        self._buf = self._buf[-7:]
                    break
                out_visible.append(self._buf[:m.start()])
                self._buf = self._buf[m.end():]
                self._in_think = True
        return "".join(out_visible), "".join(out_thought)

    def flush(self) -> tuple[str, str]:
        rem = self._buf
        self._buf = ""
        if self._in_think:
            return "", rem
        return rem, ""


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
        think_filter = _ThinkTagFilter()

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
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        # LM Studio surfaces reasoning models' thoughts in a
                        # dedicated field. Pass through as a thought event.
                        yield {"type": "thought", "delta": delta["reasoning_content"]}
                    if "content" in delta and delta["content"]:
                        text = delta["content"]
                        visible, thought = think_filter.feed(text)
                        if thought:
                            yield {"type": "thought", "delta": thought}
                        if visible:
                            full_text_parts.append(visible)
                            yield {"type": "text", "delta": visible}
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

        # flush any residual buffered text from the think filter
        vis_tail, thought_tail = think_filter.flush()
        if thought_tail:
            yield {"type": "thought", "delta": thought_tail}
        if vis_tail:
            full_text_parts.append(vis_tail)
            yield {"type": "text", "delta": vis_tail}

        yield {
            "type": "finish",
            "reason": finish_reason,
            "full_text": "".join(full_text_parts),
        }
