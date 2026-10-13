"""Receive loop tool call and turn tracking tests against a scripted fake Live session."""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from google.genai import types  # noqa: E402

from src.gemini_live.audio import AudioLoopsMixin  # noqa: E402
from src.gemini_live.chatbox import ChatboxFormattersMixin  # noqa: E402
from src.gemini_live.receive import ReceiveLoopMixin  # noqa: E402
from src.model_caps import resolve_model  # noqa: E402

SLOW = 0.3
QUICK = 0.05


class FakeConfig:
    def __init__(self, model, blocking=()):
        self.model = model
        self.model_capabilities = resolve_model(model)
        self._blocking = set(blocking)
        self.vad_mode = "auto"
        self.obs_enabled = False
        self.chatbox_page_delay = 0.2

    def tool_runs_blocking(self, name):
        return self.model_capabilities.async_tools == "none" or name in self._blocking


class FakeToolHandler:
    def __init__(self):
        self.done = {}

    def apply_response_hints(self, fr):
        return fr

    async def handle(self, fc):
        await asyncio.sleep(SLOW if fc.name.startswith("slow") else QUICK)
        self.done[fc.id] = time.monotonic()
        return types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "ok"})


class FakeSession:
    def __init__(self, script):
        self.script = list(script)
        self.tool_responses = []
        self.realtime = []
        self._hold = asyncio.Event()

    async def receive(self):
        while self.script:
            item = self.script.pop(0)
            if isinstance(item, (int, float)):
                await asyncio.sleep(item)
                continue
            yield item
        await self._hold.wait()

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(([fr.id for fr in function_responses], time.monotonic()))

    async def send_realtime_input(self, **kwargs):
        self.realtime.append(kwargs)


async def _page_slowly(pages, delay):
    await asyncio.sleep(delay * len(pages))


class Harness(ReceiveLoopMixin, AudioLoopsMixin, ChatboxFormattersMixin):
    def __init__(self, config):
        self.config = config
        self.audio = MagicMock()
        self.audio.is_music_playing.return_value = False
        self.osc = MagicMock()
        self.osc.send_chatbox_paginated.side_effect = lambda text: [text[i:i + 100] for i in range(0, len(text), 100)]
        self.osc.display_pages = AsyncMock(side_effect=_page_slowly)
        self.tool_handler = FakeToolHandler()
        self._conv_logger = MagicMock()
        self._idle_chatbox = MagicMock()
        self._emotion_system = None
        self._wanderer = None
        self._tts = None
        self._thinking_shown = False
        self._speaking = False
        self._interaction_busy = False
        self._last_model_activity = 0
        self._uses_interaction_status = not config.model_capabilities.turn_complete_is_idle
        self._tool_tasks = {}
        self._cancelled_tool_ids = set()
        self._blocking_tool_calls = 0
        self._tool_call_pending = False
        self._chatbox_task = None
        self._transcript_buffer = ""
        self._input_transcript_buffer = ""
        self._pending_finalize_task = None
        self._audio_in_queue = asyncio.Queue()
        self._playback_interrupted = False
        self._audio_gated = False
        self._audio_stream_active = False
        self._manual_vad_speaking = False
        self._last_audio_time = 0
        self._last_interaction_time = 0
        self._idle_engagement_sent = False
        self._usage_metadata = {"tool_calls": 0}
        self._reconnect_requested = False
        self._session = None

    def _schedule_user_finalize(self):
        pass

    def _check_custom_compression(self, _tokens):
        pass


def audio(turn_complete=False, status=None, text=None):
    sc = types.LiveServerContent(
        model_turn=types.Content(parts=[types.Part(inline_data=types.Blob(data=b"\x00\x01" * 100, mime_type="audio/pcm"))]),
        turn_complete=turn_complete or None,
        interaction_status=status,
        output_transcription=types.Transcription(text=text) if text else None,
    )
    return types.LiveServerMessage(server_content=sc)


def done(status=None):
    return types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True, interaction_status=status))


def tool_call(*calls):
    fcs = [types.FunctionCall(id=cid, name=name, args={}) for cid, name in calls]
    return types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=fcs))


def cancel(*ids):
    return types.LiveServerMessage(tool_call_cancellation=types.LiveServerToolCallCancellation(ids=list(ids)))


async def run_script(h, script, settle):
    session = FakeSession(script)
    h._session = session
    loop_task = asyncio.create_task(h._receive_loop(session))
    await asyncio.sleep(settle)
    return session, loop_task


async def stop(loop_task):
    loop_task.cancel()
    await asyncio.gather(loop_task, return_exceptions=True)


def test_background_tool_keeps_audio_flowing():
    async def go():
        h = Harness(FakeConfig("gemini-3.8-live"))
        session, loop_task = await run_script(h, [audio(text="checking now"), tool_call(("c1", "slowSearch")), audio()], 0.05)
        assert h._audio_in_queue.qsize() == 2, "audio around the call was dropped or stalled"
        assert not h._audio_gated and not h._tool_call_pending, "mic got shut for a background call"
        assert session.tool_responses == [], "response went out before the tool finished"
        await asyncio.sleep(SLOW + 0.1)
        assert [ids for ids, _ in session.tool_responses] == [["c1"]]
        assert h._tool_tasks == {}
        await stop(loop_task)
    asyncio.run(go())


def test_parallel_background_calls_answer_independently():
    async def go():
        h = Harness(FakeConfig("gemini-3.8-live"))
        session, loop_task = await run_script(h, [tool_call(("a", "slowSearch"), ("b", "quick"))], 0.1)
        assert [ids for ids, _ in session.tool_responses] == [["b"]], "fast result waited on the slow one"
        await asyncio.sleep(SLOW)
        assert [ids for ids, _ in session.tool_responses] == [["b"], ["a"]]
        await stop(loop_task)
    asyncio.run(go())


def test_blocking_tool_gates_mic_until_answered():
    async def go():
        h = Harness(FakeConfig("gemini-3.8-live", blocking={"getCurrentTime"}))
        h._audio_stream_active = True
        session, loop_task = await run_script(h, [audio(), tool_call(("t1", "getCurrentTime"))], 0.01)
        assert h._tool_call_pending and h._audio_gated
        assert {"audio_stream_end": True} in session.realtime
        await asyncio.sleep(QUICK + 0.05)
        assert [ids for ids, _ in session.tool_responses] == [["t1"]]
        assert not h._tool_call_pending and not h._audio_gated
        assert h._audio_in_queue.qsize() == 1, "3.8 blocking call should not throw away queued speech"
        await stop(loop_task)
    asyncio.run(go())


def test_cancelled_call_result_is_not_sent():
    async def go():
        h = Harness(FakeConfig("gemini-3.8-live"))
        session, loop_task = await run_script(h, [tool_call(("c1", "slowSearch")), 0.05, cancel("c1")], SLOW + 0.15)
        assert "c1" in h.tool_handler.done, "tool should still run to completion"
        assert session.tool_responses == []
        assert h._cancelled_tool_ids == set()
        await stop(loop_task)
    asyncio.run(go())


def test_extended_thinking_stays_busy_until_idle_with_mic_open():
    async def go():
        h = Harness(FakeConfig("gemini-3.8-live-extended-thinking"))
        session, loop_task = await run_script(
            h, [audio(text="let me look"), done(types.InteractionStatus.IN_PROGRESS), tool_call(("c1", "slowSearch"))], 0.05)
        assert h._interaction_busy and not h._speaking
        assert not h._audio_gated, "mic should stay open while it works in the background"
        await asyncio.sleep(SLOW + 0.1)
        session.script += [audio(text="found it"), done(types.InteractionStatus.IDLE)]
        await stop(loop_task)
        loop_task = asyncio.create_task(h._receive_loop(session))
        await asyncio.sleep(0.05)
        assert not h._interaction_busy and not h._speaking
        await stop(loop_task)
    asyncio.run(go())


def test_plain_turn_complete_ends_turn_without_status():
    async def go():
        h = Harness(FakeConfig("gemini-3.8-live"))
        _session, loop_task = await run_script(h, [audio(), done()], 0.05)
        assert not h._interaction_busy and not h._speaking
        await stop(loop_task)
    asyncio.run(go())


def test_legacy_blocking_call_still_drops_stale_audio():
    async def go():
        h = Harness(FakeConfig("gemini-3.1-flash-live-preview"))
        session, loop_task = await run_script(h, [audio(), tool_call(("c1", "anything"))], 0.01)
        assert h._audio_in_queue.qsize() == 0 and h._tool_call_pending
        await asyncio.sleep(QUICK + 0.05)
        assert [ids for ids, _ in session.tool_responses] == [["c1"]]
        assert not h._tool_call_pending
        await stop(loop_task)
    asyncio.run(go())


def test_chatbox_paging_does_not_hold_up_the_next_turn():
    async def go():
        h = Harness(FakeConfig("gemini-3.8-live"))
        long_text = "word " * 120
        _session, loop_task = await run_script(h, [audio(text=long_text), done(), audio(text="next")], 0.05)
        assert h._audio_in_queue.qsize() == 2, "next turn waited on chatbox paging"
        await stop(loop_task)
    asyncio.run(go())


def test_old_session_tool_finishing_late_leaves_new_gate_alone():
    async def go():
        h = Harness(FakeConfig("gemini-3.8-live", blocking={"slowLookup"}))
        old, loop_task = await run_script(h, [tool_call(("c1", "slowLookup"))], 0.05)
        await stop(loop_task)
        h._session = object()
        h._blocking_tool_calls = 1
        h._tool_call_pending = True
        await asyncio.sleep(SLOW)
        assert old.tool_responses == []
        assert h._tool_call_pending and h._blocking_tool_calls == 1
    asyncio.run(go())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
