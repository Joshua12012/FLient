"""
Run Flower (flwr) in a subprocess and keep it reachable on the gRPC port.

Flower exits after `num_rounds` completes, so nothing listens on :8080 until we
spawn a new process. FastAPI calls `ensure_running()` periodically and from
`/api/metrics/reset` / `/api/fl/restart` so a second training session works
without restarting the whole server.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

# `web/server` (directory that contains the `app` package).
_SERVER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROC: Optional[subprocess.Popen] = None
_LAST_START_TS: float = 0.0


def _build_cmd() -> list[str]:
    host = os.environ.get("FL_HOST", "0.0.0.0")
    port = os.environ.get("FL_PORT", "8080")
    rounds = os.environ.get("FL_ROUNDS", "10")
    min_clients = os.environ.get("FL_MIN_CLIENTS", "2")
    tier = os.environ.get("FL_TIER", "standard")
    return [
        sys.executable,
        "-m",
        "app.fl.server_runner",
        "--host",
        str(host),
        "--port",
        str(port),
        "--rounds",
        str(rounds),
        "--min-clients",
        str(min_clients),
        "--tier",
        str(tier),
    ]


def is_running() -> bool:
    return _PROC is not None and _PROC.poll() is None


def stop(timeout_s: float = 12.0) -> None:
    global _PROC
    if _PROC is None:
        return
    if _PROC.poll() is not None:
        _PROC = None
        return
    _PROC.terminate()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _PROC.poll() is not None:
            break
        time.sleep(0.2)
    if _PROC.poll() is None:
        _PROC.kill()
    _PROC = None


def start() -> subprocess.Popen:
    """Start Flower subprocess (replaces dead process). Caller should `stop()` first if restarting."""
    global _PROC, _LAST_START_TS
    cmd = _build_cmd()
    print(f"[FlowerSupervisor] Popen: {' '.join(cmd)}")
    _PROC = subprocess.Popen(cmd, cwd=_SERVER_ROOT)
    _LAST_START_TS = time.time()
    return _PROC


def ensure_running() -> Dict[str, Any]:
    """If Flower is not running, start it. Returns status dict for /api/fl/*."""
    global _PROC
    if is_running():
        return status()
    start()
    return status()


def restart() -> Dict[str, Any]:
    """Hard restart Flower (new federated run from initial weights)."""
    stop()
    start()
    return status()


def status() -> Dict[str, Any]:
    alive = is_running()
    pid = _PROC.pid if _PROC is not None else None
    exit_code = _PROC.poll() if _PROC is not None else None
    return {
        "flower_alive": alive,
        "pid": pid,
        "exit_code": exit_code,
        "grpc": f"{os.environ.get('FL_HOST', '0.0.0.0')}:{os.environ.get('FL_PORT', '8080')}",
        "tier": os.environ.get("FL_TIER", "standard"),
        "rounds": int(os.environ.get("FL_ROUNDS", "10")),
        "min_clients": int(os.environ.get("FL_MIN_CLIENTS", "2")),
        "dataset": os.environ.get("FL_DATASET", "emnist"),
        "last_start_ts": _LAST_START_TS,
    }
