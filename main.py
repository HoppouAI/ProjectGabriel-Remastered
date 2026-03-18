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

    # Wanderer for autonomous exploration (lazy import to skip heavy deps when disabled)
    wanderer = None
    if config.wanderer_enabled:
        from src.wanderer import Wanderer
        wanderer = Wanderer(config, osc)
        wanderer.preload()

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

    # Instance monitor for player list (VRChat log parsing)
    from src.instance_monitor import InstanceMonitor
    instance_monitor = InstanceMonitor()
    instance_monitor.start()
    session.tool_handler.instance_monitor = instance_monitor

    # VRChat API for avatar switching (background login)
    vrchat_api = None
    if config.vrchat_api_username:
        from src.vrchatapi import VRChatAPI
        vrchat_api = VRChatAPI(config)
        session.tool_handler.vrchat_api = vrchat_api

        async def _bg_login():
            try:
                ok = await vrchat_api.ensure_logged_in()
                if ok:
                    logger.info("VRChat API authenticated successfully")
                    if not VRChatAPI.friends_cache_fresh():
                        logger.info("Friends cache is stale (>4h) or missing, fetching...")
                        await vrchat_api.fetch_and_cache_friends()
                    else:
                        logger.info("Friends cache is fresh, skipping fetch")
                else:
                    logger.warning("VRChat API login failed -- avatar switching may not work")
            except Exception as e:
                logger.error(f"VRChat API background login error: {e}")

        asyncio.create_task(_bg_login())

    # Wire wanderer into tool handler and session
    if wanderer:
        session.tool_handler.wanderer = wanderer
        session._wanderer = wanderer
        if face_tracker:
            wanderer._face_tracker_ref = face_tracker
        wanderer._emotion_system_ref = get_emotion_system()

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
