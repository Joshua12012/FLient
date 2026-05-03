"""
HTTP endpoints for round metrics (backed by data/round_metrics.json).

GET /api/metrics/rounds    -> JSON list[RoundMetric]
GET /api/metrics/summary   -> JSON summary stats
GET /api/metrics/json      -> full file blob (reloads from disk; Flower runs in subprocess)
GET /api/metrics/chart.png -> tiny placeholder PNG (no matplotlib)
POST /api/metrics/reset    -> Clear metrics + restart Flower
"""

from dataclasses import asdict

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from app.fl.flower_supervisor import restart as flower_restart
from app.fl.metrics import get_store

metrics_router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@metrics_router.get("/rounds")
def list_rounds():
    return JSONResponse([asdict(r) for r in get_store().rounds()])


@metrics_router.get("/summary")
def summary():
    return JSONResponse(get_store().summary())


@metrics_router.get("/json")
def metrics_json_file():
    """Same content as `web/server/data/round_metrics.json` (reloaded from disk)."""
    return JSONResponse(get_store().full_blob())


@metrics_router.get("/chart.png")
def chart_png():
    png = get_store().render_chart_png()
    return Response(content=png, media_type="image/png")


@metrics_router.post("/reset")
def reset_metrics():
    """
    Clear stored round metrics and restart the Flower subprocess.

    Flower exits after `num_rounds`; without a restart, gRPC :8080 refuses new
    connections — this reset path is what the Gradio "New session" button uses.
    """
    get_store().reset()
    flower_restart()
    return {"status": "ok", "message": "metrics cleared; Flower restarted for a new run"}
