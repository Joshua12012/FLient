"""
FastAPI Federated Learning Server (Flower-powered).

Architecture:
- FastAPI on port 8000 serves HTTP routes:
    * /api/models/*  - download adaptive Keras / TFLite snapshots
    * /api/metrics/* - per-round metrics JSON (see data/round_metrics.json)
    * /                / dashboard / health
- Flower (flwr) gRPC server on port 8080 (subprocess + watchdog inside `lifespan`):
    * AdaptiveFedAvg strategy aggregates client updates and records metrics
    * Snapshots are saved per round to web/server/models/global_<tier>.{weights.h5,tflite}

Clients (web/client/fl_gradio_app.py) connect via Flower's gRPC API and additionally
poll the FastAPI metrics + models endpoints for live charts and adaptive downloads.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Allow `import app.*` when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.fl.flower_supervisor import ensure_running, stop as flower_stop  # noqa: E402
from app.routers.fl_process_router import fl_process_router  # noqa: E402
from app.routers.metrics_router import metrics_router  # noqa: E402
from app.routers.model_router import model_router  # noqa: E402


_WATCHDOG_TASK: Optional[asyncio.Task] = None


async def _flower_watchdog() -> None:
    """If Flower exits after completing all rounds, bring gRPC back without a full server restart."""
    while True:
        await asyncio.sleep(4.0)
        try:
            ensure_running()
        except Exception as exc:
            print(f"[FlowerSupervisor] watchdog error: {exc}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

# Module-level config filled by `main()` before `uvicorn.run` so the lifespan
# function can pick it up after the reload child process re-imports this module.
FL_HOST = os.environ.get("FL_HOST", "0.0.0.0")
FL_PORT = int(os.environ.get("FL_PORT", "8080"))
FL_ROUNDS = int(os.environ.get("FL_ROUNDS", "10"))
FL_MIN_CLIENTS = int(os.environ.get("FL_MIN_CLIENTS", "2"))
FL_TIER = os.environ.get("FL_TIER", "standard")
FL_DATASET = os.environ.get("FL_DATASET", "emnist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bring Flower up on startup and terminate it on shutdown."""
    global _WATCHDOG_TASK
    print("[Server] Starting FastAPI + Flower...")
    ensure_running()
    _WATCHDOG_TASK = asyncio.create_task(_flower_watchdog())
    yield
    print("[Server] Shutting down...")
    if _WATCHDOG_TASK is not None:
        _WATCHDOG_TASK.cancel()
        try:
            await _WATCHDOG_TASK
        except asyncio.CancelledError:
            pass
        _WATCHDOG_TASK = None
    flower_stop()


app = FastAPI(
    title="Federated Learning Server (Flower)",
    description=(
        "FastAPI + Flower (flwr) FL server. "
        "Trains an EMNIST byclass CNN across Gradio clients and exposes "
        "per-round metrics + adaptive model downloads."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metrics_router)
app.include_router(model_router)
app.include_router(fl_process_router)

# Static dir for serving raw artifacts if needed.
os.makedirs("models", exist_ok=True)
app.mount("/models", StaticFiles(directory="models"), name="models")


@app.get("/")
def root():
    return {
        "service": "Federated Learning Server",
        "version": "2.0.0",
        "fl_engine": "flower (flwr)",
        "fl_grpc": f"{FL_HOST}:{FL_PORT}",
        "dataset": FL_DATASET,
        "metrics_json_file": "data/round_metrics.json (also GET /api/metrics/json)",
        "endpoints": {
            "docs": "/docs",
            "fl_status": "/api/fl/status",
            "fl_ensure": "/api/fl/ensure",
            "fl_restart": "/api/fl/restart",
            "metrics_summary": "/api/metrics/summary",
            "metrics_rounds": "/api/metrics/rounds",
            "metrics_json": "/api/metrics/json",
            "metrics_chart": "/api/metrics/chart.png",
            "tiers": "/api/models/tiers",
            "weights": "/api/models/download/{tier}/weights",
            "tflite": "/api/models/download/{tier}/tflite",
        },
    }


@app.get("/health")
def health():
    from app.fl.flower_supervisor import is_running

    return {
        "status": "healthy",
        "flower_alive": is_running(),
        "fl_grpc": f"{FL_HOST}:{FL_PORT}",
        "fl_tier": FL_TIER,
        "dataset": FL_DATASET,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Tiny HTML dashboard with the live chart and a refresh button."""
    return """
    <!DOCTYPE html>
    <html><head><title>FL Dashboard</title>
      <style>
        body { font-family: -apple-system, sans-serif; margin: 32px; background: #0f172a; color: #e2e8f0; }
        a { color: #60a5fa; }
        img { max-width: 100%; border: 1px solid #334155; border-radius: 8px; }
        pre { background: #1e293b; padding: 12px; border-radius: 8px; overflow-x: auto; }
      </style>
      <script>
        async function refresh() {
          const sum = await (await fetch('/api/metrics/summary')).json();
          document.getElementById('summary').innerText = JSON.stringify(sum, null, 2);
          document.getElementById('chart').src = '/api/metrics/chart.png?ts=' + Date.now();
        }
        setInterval(refresh, 3000);
        window.addEventListener('load', refresh);
      </script>
    </head><body>
      <h1>Federated Learning Dashboard</h1>
      <p><a href="/docs">API docs</a> &middot; <a href="/api/metrics/rounds">/rounds</a></p>
      <h2>Summary</h2>
      <pre id="summary">loading...</pre>
      <h2>Chart</h2>
      <img id="chart" src="/api/metrics/chart.png" />
    </body></html>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description="FL Server (FastAPI + Flower)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--fl-host", default="0.0.0.0", help="Flower gRPC host")
    parser.add_argument("--fl-port", type=int, default=8080, help="Flower gRPC port")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Min clients per round")
    parser.add_argument(
        "--tier",
        default="standard",
        choices=["lite", "standard", "full"],
        help="Tier of the global model",
    )
    parser.add_argument(
        "--dataset",
        default="emnist",
        choices=["emnist", "fashion_mnist", "kmnist"],
        help="Vision dataset (28x28 grayscale); all clients/server must use the same",
    )
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload")
    args = parser.parse_args()

    # Pass FL config to child process via env vars so reload still works.
    os.environ["FL_HOST"] = args.fl_host
    os.environ["FL_PORT"] = str(args.fl_port)
    os.environ["FL_ROUNDS"] = str(args.rounds)
    os.environ["FL_MIN_CLIENTS"] = str(args.min_clients)
    os.environ["FL_TIER"] = args.tier
    os.environ["FL_DATASET"] = args.dataset

    print("=" * 60)
    print(f"FastAPI HTTP : http://{args.host}:{args.port}")
    print(f"Flower gRPC  : {args.fl_host}:{args.fl_port}")
    print(
        f"Rounds       : {args.rounds}   Min clients: {args.min_clients}   "
        f"Tier: {args.tier}   Dataset: {args.dataset}"
    )
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
