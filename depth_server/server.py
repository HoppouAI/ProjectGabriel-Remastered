"""
Depth estimation server for ProjectGabriel wanderer.

Runs a depth model on a dedicated GPU and exposes it via HTTP.
The wanderer sends frames and receives depth maps, offloading
inference from the VRChat machine.

Usage:
    set DEPTH_API_KEY=your-secret-key

    # Run directly
    python server.py

    # Or with uv
    uv run server.py

    # With options
    python server.py --model depth-anything-v2-base --port 8780
"""

import argparse
import hmac
import io
import logging
import os
import secrets
import sys
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

logger = logging.getLogger("depth_server")

MODELS = {
    "depth-anything-v2-small": {
        "repo": "depth-anything/Depth-Anything-V2-Small-hf",
        "invert": False,
        "default_size": 392,
    },
    "depth-anything-v2-base": {
        "repo": "depth-anything/Depth-Anything-V2-Base-hf",
        "invert": False,
        "default_size": 518,
    },
    "dpt-large": {
        "repo": "Intel/dpt-large",
        "invert": False,
        "default_size": 384,
    },
}

OUT_W = 384
OUT_H = 288

app = FastAPI(title="Depth Server", docs_url=None, redoc_url=None)

_model = None
_transform = None
_device = None
_invert = False
_api_key = None
_use_fp16 = False


def load_model(model_key: str, input_size: int = 0, fp16: bool = True):
    global _model, _transform, _device, _invert, _use_fp16

    spec = MODELS.get(model_key)
    if not spec:
        raise ValueError(f"Unknown model: {model_key}. Options: {list(MODELS.keys())}")

    repo = spec["repo"]
    _invert = spec["invert"]
    size = input_size or spec["default_size"]
    size = max(14, (size // 14) * 14)

    logger.info(f"Loading {model_key} from {repo} (input {size}x{size})")

    try:
        _transform = AutoImageProcessor.from_pretrained(
            repo, use_fast=True,
            size={"height": size, "width": size},
        )
    except Exception:
        from transformers import DPTImageProcessor
        _transform = DPTImageProcessor(
            do_resize=True,
            size={"height": size, "width": size},
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

    _model = AutoModelForDepthEstimation.from_pretrained(repo)

    if torch.cuda.is_available():
        _device = "cuda"
        _model.to("cuda")
        if fp16:
            _model.half()
            _use_fp16 = True
            logger.info(f"Model on CUDA FP16 ({torch.cuda.get_device_name(0)})")
        else:
            logger.info(f"Model on CUDA ({torch.cuda.get_device_name(0)})")
    else:
        _device = "cpu"
        _use_fp16 = False
        logger.info("Model on CPU (will be slow)")

    _model.eval()
    logger.info("Model ready")


def run_depth(image: Image.Image) -> np.ndarray:
    """Run depth estimation. Returns 0-1 map where higher = closer (obstacle)."""
    inputs = _transform(images=image, return_tensors="pt")
    if _device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        if _use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

    with torch.no_grad():
        depth = _model(**inputs).predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1).float(),
        size=(OUT_H, OUT_W),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = depth.cpu().numpy()
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_np)

    if _invert:
        depth_norm = 1.0 - depth_norm

    return depth_norm


@app.post("/depth")
async def estimate_depth(
    file: UploadFile = File(...),
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    if not hmac.compare_digest(x_api_key, _api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    t0 = time.perf_counter()
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    depth_map = run_depth(img)

    depth_uint8 = (np.clip(depth_map, 0, 1) * 255).astype(np.uint8)
    _, png = cv2.imencode(".png", depth_uint8)

    ms = (time.perf_counter() - t0) * 1000
    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={"X-Inference-Ms": f"{ms:.0f}"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth estimation server")
    parser.add_argument("--model", default="depth-anything-v2-small", choices=list(MODELS.keys()))
    parser.add_argument("--input-size", type=int, default=0, help="Model input resolution (0 = model default)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--api-key", default=None, help="API key (overrides DEPTH_API_KEY env var)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys.txt")

    # Priority: CLI arg > env var > keys.txt file
    _api_key = args.api_key or os.environ.get("DEPTH_API_KEY")
    if not _api_key:
        if os.path.exists(keys_path):
            with open(keys_path) as f:
                _api_key = f.read().strip()
        if not _api_key:
            _api_key = secrets.token_urlsafe(32)
            with open(keys_path, "w") as f:
                f.write(_api_key)
            logger.info(f"Generated new API key and saved to {keys_path}")

    logger.info(f"API key: {_api_key}")

    load_model(args.model, args.input_size, fp16=not args.no_fp16)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
