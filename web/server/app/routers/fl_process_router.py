"""
Flower process control (no training logic — only supervises the gRPC subprocess).

GET  /api/fl/status  -> is Flower alive, pid, tier, rounds
POST /api/fl/ensure  -> start Flower if it exited (e.g. after num_rounds finished)
POST /api/fl/restart -> terminate + start Flower (fresh run)
"""

from fastapi import APIRouter

from app.fl.flower_supervisor import ensure_running, restart, status

fl_process_router = APIRouter(prefix="/api/fl", tags=["flower"])


@fl_process_router.get("/status")
def fl_status():
    return status()


@fl_process_router.post("/ensure")
def fl_ensure():
    return ensure_running()


@fl_process_router.post("/restart")
def fl_restart():
    return restart()
