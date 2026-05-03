"""
Adaptive model distribution.

Endpoints:
- GET /api/models/tiers                 -> list available tiers + parameter counts
- GET /api/models/download/{tier}/weights  -> latest .weights.h5 (best accuracy)
- GET /api/models/download/{tier}/tflite   -> latest float16 .tflite (small + fast)

The artifacts come from `app.fl.inference` and are refreshed every FL round
by `AdaptiveFedAvg.aggregate_fit`.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..fl import inference
from ..fl.model import TIER_CONFIGS, build_model

model_router = APIRouter(prefix="/api/models", tags=["models"])


@model_router.get("/tiers")
def list_tiers():
    """List the three tiers with parameter counts and descriptions."""
    out = {}
    for name in TIER_CONFIGS:
        m = build_model(name)
        out[name] = {
            "tier": name,
            "description": TIER_CONFIGS[name]["description"],
            "parameters": int(m.count_params()),
            "size_kb": round(m.count_params() * 4 / 1024, 2),
        }
    return JSONResponse(out)


@model_router.get("/download/{tier}/weights")
def download_weights(tier: str):
    """Serve the latest Keras .weights.h5 for a tier (full precision)."""
    if tier not in TIER_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown tier '{tier}'")
    path = inference.weights_path(tier)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No global weights yet for tier '{tier}'. Run at least one FL round.",
        )
    return FileResponse(path, media_type="application/octet-stream", filename=os.path.basename(path))


@model_router.get("/download/{tier}/tflite")
def download_tflite(tier: str):
    """Serve the latest float16-quantized TFLite for a tier (mobile / lite tier)."""
    if tier not in TIER_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown tier '{tier}'")
    path = inference.tflite_path(tier)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No TFLite snapshot yet for tier '{tier}'. Run at least one FL round.",
        )
    return FileResponse(path, media_type="application/octet-stream", filename=os.path.basename(path))


@model_router.get("/recommend")
def recommend_artifact(tier: str = "standard", battery: float = 1.0, memory_mb: float = 1024):
    """
    Adaptive deployment recommendation:
    - low battery / low memory / lite tier -> TFLite float16
    - otherwise                            -> Keras weights
    """
    use_tflite = (
        tier == "lite"
        or battery < 0.25
        or memory_mb < 512
    )
    return {
        "tier": tier,
        "artifact": "tflite" if use_tflite else "weights",
        "url": f"/api/models/download/{tier}/{'tflite' if use_tflite else 'weights'}",
    }
