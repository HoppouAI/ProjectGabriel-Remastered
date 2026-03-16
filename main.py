import asyncio
import logging

# Import tracker FIRST — its module-level code pre-initialises bettercam
# via DXGI Desktop Duplication BEFORE any CUDA library loads.
# If CUDA loads first, DXGI fails on hybrid-GPU systems.
# Importing is safe even when tracker is disabled in config.
from src.tracker import PlayerTracker

from src.config import Config
from src.audio import AudioManager
from src.vrchat import VRChatOSC
from src.personalities import PersonalityManager
from src.gemini_live import GeminiLiveSession
from src.emotions import get_emotion_system
from src.memory import memory_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gabriel")

# Suppress the known CPython 3.12 Windows ProactorEventLoop assertion error
# This fires during pipe transport cleanup and is harmless
class _ProactorAssertFilter(logging.Filter):
    def filter(self, record):
        return not (record.name == "asyncio" and "_loop_writing" in str(record.msg))

logging.getLogger("asyncio").addFilter(_ProactorAssertFilter())


def setup_control_server(session, audio, personality, memory, get_emotion_fn):
    """Setup the control panel shared state and return a uvicorn Server."""
    try:
        from control_server import app, shared_state
        import uvicorn
        
        shared_state["session"] = session
        shared_state["audio_mgr"] = audio
        shared_state["personality_mgr"] = personality
        shared_state["memory_mgr"] = memory
        shared_state["get_emotion_fn"] = get_emotion_fn
        logger.info("Starting control panel on http://localhost:8766")
        config = uvicorn.Config(app, host="0.0.0.0", port=8766, log_level="warning")
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # Don't override main app's signals
        return server
    except ImportError:
        logger.warning("Control server not available (missing dependencies)")
        return None
    except Exception as e:
        logger.error(f"Control server setup error: {e}")
        return None


async def main():
    loop = asyncio.get_running_loop()
    _orig_handler = loop.get_exception_handler()
    def _suppress_proactor_write_assert(loop, context):
        exc = context.get("exception")
        if isinstance(exc, AssertionError) and "_loop_writing" in str(context.get("handle", "")):
            return
        if _orig_handler:
            _orig_handler(loop, context)
        else:
            loop.default_exception_handler(context)
    loop.set_exception_handler(_suppress_proactor_write_assert)

    config = Config()
    audio = AudioManager(config)
    osc = VRChatOSC(config)
    tracker = PlayerTracker(config, osc) if config.tracker_enabled else None
    if tracker:
        tracker.preload()  # async background model load + warmup
        if config.vision_debug:
            from vision_server import run_vision_server
            tracker._vision_debug = True
            run_vision_server(port=config.vision_debug_port, tracker=tracker)

    # Face tracker for looking at people (lazy import to skip heavy deps when disabled)
    face_tracker = None
    if config.face_tracker_enabled:
        from src.face_tracker import FaceTracker
        face_tracker = FaceTracker(config, osc)
        face_tracker.preload()

    personality = PersonalityManager()

    # External TTS provider (optional - when tts.provider != "gemini")
    tts_provider = None
    if config.tts_qwen3_enabled:
        from src.tts import QwenTTSProvider
        tts_provider = QwenTTSProvider(config)
        tts_provider.start()
        logger.info("Using Qwen3 TTS provider (Gemini audio will be discarded)")
    elif config.tts_hoppou_enabled:
        from src.tts import HoppouTTSProvider
        tts_provider = HoppouTTSProvider(config)
        tts_provider.start()
        logger.info("Using Hoppou TTS provider (Gemini audio will be discarded)")
    elif config.tts_chirp3_hd_enabled:
        from src.tts import Chirp3HDTTSProvider
        tts_provider = Chirp3HDTTSProvider(config)
        tts_provider.start()
        logger.info("Using Chirp 3 HD TTS provider (Gemini audio will be discarded)")

    session = GeminiLiveSession(config, audio, osc, tracker, personality, tts_provider)

    # Wire face tracker speaking callback and start
    if face_tracker:
        face_tracker.set_speaking_callback(lambda: session._speaking)
        if tracker:
            face_tracker.set_player_tracker(tracker)
        face_tracker.start()

    logger.info("ProjectGabriel starting...")
    logger.info(f"Model: {config.model}")
    logger.info(f"Music dir: {config.music_dir}")
    logger.info(f"OSC → {config.osc_ip}:{config.osc_port}")

    # Start control panel as async task in same event loop
    control_server = setup_control_server(session, audio, personality, memory_system, get_emotion_system)
    if control_server:
        asyncio.create_task(control_server.serve())

    while True:
        try:
            await session.run()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Session crashed: {e}")
            logger.info("Restarting session in 3 seconds...")
            await asyncio.sleep(3)
            continue
    
    # Cleanup only happens on KeyboardInterrupt
    if tts_provider:
        tts_provider.stop()
    if face_tracker:
        face_tracker.stop()
    if tracker:
        tracker.active = False
    emotion = get_emotion_system()
    if emotion:
        emotion.stop()
    audio.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
