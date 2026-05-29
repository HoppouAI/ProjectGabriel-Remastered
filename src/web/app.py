"""FastAPI app for the Gabriel control panel. Wires all routers together."""
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .shared import shared_state
from .routes import (
    core_router,
    mapping_router,
    memory_router,
    music_router,
    pages_router,
    start_music_broadcast,
    start_state_broadcast,
    vision_router,
    vrchat_router,
    ws_router,
)

STATIC_DIR = Path(__file__).parent.parent.parent / "webui"

app = FastAPI(title="Control Panel")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(pages_router)
app.include_router(core_router)
app.include_router(music_router)
app.include_router(memory_router)
app.include_router(vrchat_router)
app.include_router(mapping_router)
app.include_router(vision_router)
app.include_router(ws_router)


@app.on_event("startup")
async def _on_startup():
    start_music_broadcast()
    start_state_broadcast()


def run_control_server(host: str = "0.0.0.0", port: int = 8766):
    cfg = shared_state.get("config")
    name = cfg.app_name if cfg else "Control Panel"
    print(f"{name} Control Panel")
    print(f"Open http://localhost:{port} in your browser")
    uvicorn.run(app, host=host, port=port, log_level="warning")
