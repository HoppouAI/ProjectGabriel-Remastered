"""Core session control endpoints: state, mute, reconnect, text input, personality, emotion."""
from fastapi import APIRouter, HTTPException

from ..models import EmotionInput, PersonalityInput, TextInput
from ..shared import (
    SESSION_HANDLE_FILE,
    SEVEN_ZIP_PATH,
    add_console_log,
    broadcast_state,
    console_logs,
    get_full_state,
    shared_state,
)

router = APIRouter()


@router.get("/api/state")
async def get_state():
    return get_full_state()


@router.get("/api/console-logs")
async def get_console_logs():
    return list(console_logs)


@router.get("/api/sevenzip-status")
async def sevenzip_status():
    return {"available": SEVEN_ZIP_PATH is not None, "path": SEVEN_ZIP_PATH}


@router.post("/api/reconnect")
async def reconnect():
    session = shared_state.get("session")
    if session and hasattr(session, "request_reconnect"):
        session.request_reconnect()
        return {"message": "Reconnect requested"}
    return {"message": "No active session to reconnect"}


@router.post("/api/clear-session")
async def clear_session():
    if SESSION_HANDLE_FILE.exists():
        SESSION_HANDLE_FILE.unlink()
    session = shared_state.get("session")
    if session:
        if hasattr(session, "_session_handle"):
            session._session_handle = None
        session.request_reconnect()
        return {"message": "Session cleared and reconnect requested"}
    return {"message": "Session handle cleared"}


@router.post("/api/toggle-mute")
async def toggle_mute():
    session = shared_state.get("session")
    if session and hasattr(session, "_mic_muted"):
        session._mic_muted = not session._mic_muted
        shared_state["mic_muted"] = session._mic_muted
        if hasattr(session, "osc") and session.osc:
            session.osc.toggle_voice()
    else:
        shared_state["mic_muted"] = not shared_state.get("mic_muted", False)
    await broadcast_state()
    return {"muted": shared_state["mic_muted"]}


@router.post("/api/send-text")
async def send_text(data: TextInput):
    session = shared_state.get("session")
    if not session:
        raise HTTPException(status_code=400, detail="No active session")
    if hasattr(session, "send_text"):
        await session.send_text(data.text)
        return {"message": "Text sent"}
    raise HTTPException(status_code=400, detail="Session does not support text input")


@router.post("/api/send-system-instruction")
async def send_system_instruction(data: TextInput):
    session = shared_state.get("session")
    if not session:
        raise HTTPException(status_code=400, detail="No active session")
    if not hasattr(session, "_session") or not session._session:
        raise HTTPException(status_code=400, detail="No active live session")
    try:
        from google.genai import types
        await session.send_client_content_safe(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"SYSTEM INSTRUCTION: {data.text}")],
            ),
            turn_complete=True,
        )
        add_console_log("info", f"System instruction sent: {data.text[:80]}")
        return {"message": "System instruction sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/switch-personality")
async def switch_personality(data: PersonalityInput):
    personality_mgr = shared_state.get("personality_mgr")
    session = shared_state.get("session")
    if not personality_mgr:
        raise HTTPException(status_code=400, detail="Personality manager not available")
    try:
        result = personality_mgr.switch(data.personality)
        if "personality_prompt" in result and session and hasattr(session, "_session") and session._session:
            from google.genai import types
            await session.send_client_content_safe(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=result["personality_prompt"])],
                ),
                turn_complete=True,
            )
        await broadcast_state()
        return {k: v for k, v in result.items() if k != "personality_prompt"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/trigger-emotion")
async def trigger_emotion(data: EmotionInput):
    get_emotion_fn = shared_state.get("get_emotion_fn")
    emotion_mgr = get_emotion_fn() if get_emotion_fn else None
    if not emotion_mgr:
        raise HTTPException(status_code=400, detail="Emotion manager not available")
    try:
        return emotion_mgr.play_emotion(data.emotion)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
