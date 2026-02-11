import asyncio
import logging
import threading
from src.config import Config
from src.audio import AudioManager
from src.vrchat import VRChatOSC
from src.tracker import YOLOTracker
from src.personalities import PersonalityManager
from src.gemini_live import GeminiLiveSession
from src.emotions import get_emotion_system
from src.memory import memory_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gabriel")


def start_control_server(session, audio, personality, memory, get_emotion_fn):
    """Start the control panel server in a separate thread."""
    try:
        from control_server import app, shared_state, run_control_server
        import uvicorn
        
        shared_state["session"] = session
        shared_state["audio_mgr"] = audio
        shared_state["personality_mgr"] = personality
        shared_state["memory_mgr"] = memory
        # Store reference to getter function since emotion system is initialized later
        shared_state["get_emotion_fn"] = get_emotion_fn
        logger.info("Starting control panel on http://localhost:8766")
        uvicorn.run(app, host="0.0.0.0", port=8766, log_level="warning")
    except ImportError:
        logger.warning("Control server not available (missing dependencies)")
    except Exception as e:
        logger.error(f"Control server error: {e}")


async def main():
    config = Config()
    audio = AudioManager(config)
    osc = VRChatOSC(config)
    tracker = YOLOTracker(config)
    personality = PersonalityManager()
    session = GeminiLiveSession(config, audio, osc, tracker, personality)

    logger.info("ProjectGabriel starting...")
    logger.info(f"Model: {config.model}")
    logger.info(f"Music dir: {config.music_dir}")
    logger.info(f"OSC → {config.osc_ip}:{config.osc_port}")

    # Start control panel in background thread
    control_thread = threading.Thread(
        target=start_control_server, 
        args=(session, audio, personality, memory_system, get_emotion_system), 
        daemon=True
    )
    control_thread.start()

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
    tracker.active = False
    emotion = get_emotion_system()
    if emotion:
        emotion.stop()
    audio.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
