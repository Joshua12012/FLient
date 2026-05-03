"""
Per-round metrics persisted to **web/server/data/round_metrics.json**.

The Flower strategy runs in a **subprocess** and writes this file. The FastAPI
process must **reload from disk** before serving HTTP metrics (see `reload_from_disk`).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

_DEFAULT_STORE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "round_metrics.json",
)

# 1×1 PNG — `/api/metrics/chart.png` returns this so we do not depend on matplotlib.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@dataclass
class RoundMetric:
    round: int
    accuracy: float = 0.0
    avg_loss: float = 0.0
    samples_per_sec: float = 0.0
    mem_mb: float = 0.0
    train_time_s: float = 0.0
    comm_time_s: float = 0.0
    comm_bytes: int = 0
    num_clients: int = 0
    tier: str = "standard"
    timestamp: float = field(default_factory=lambda: time.time())


class MetricsStore:
    def __init__(self, path: str = _DEFAULT_STORE):
        self.path = path
        self._lock = threading.RLock()
        self._rounds: List[RoundMetric] = []
        self._time_to_converge: Optional[float] = None
        self._target_acc: float = 0.70
        self._first_round_time: Optional[float] = None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.reload_from_disk()

    def reload_from_disk(self) -> None:
        """Re-read JSON from disk (authoritative when Flower runs in another process)."""
        with self._lock:
            if not os.path.exists(self.path):
                self._rounds = []
                return
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                self._rounds = [RoundMetric(**r) for r in blob.get("rounds", [])]
                self._time_to_converge = blob.get("time_to_converge")
                ta = blob.get("target_acc")
                if ta is not None:
                    self._target_acc = float(ta)
            except Exception as exc:
                print(f"[metrics] reload_from_disk failed: {exc}")

    def _persist(self) -> None:
        blob = {
            "rounds": [asdict(r) for r in self._rounds],
            "time_to_converge": self._time_to_converge,
            "target_acc": self._target_acc,
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        os.replace(tmp, self.path)

    def record(self, metric: RoundMetric) -> None:
        with self._lock:
            if self._first_round_time is None:
                self._first_round_time = metric.timestamp
            self._rounds.append(metric)
            if (
                self._time_to_converge is None
                and metric.accuracy >= self._target_acc
                and self._first_round_time is not None
            ):
                self._time_to_converge = metric.timestamp - self._first_round_time
            self._persist()

    def reset(self) -> None:
        with self._lock:
            self._rounds.clear()
            self._time_to_converge = None
            self._first_round_time = None
            self._persist()

    def set_target(self, acc: float) -> None:
        with self._lock:
            self._target_acc = float(acc)
            self._persist()

    def rounds(self) -> List[RoundMetric]:
        self.reload_from_disk()
        with self._lock:
            return list(self._rounds)

    def summary(self) -> Dict:
        self.reload_from_disk()
        with self._lock:
            rounds = list(self._rounds)
            if not rounds:
                return {
                    "rounds_completed": 0,
                    "final_accuracy": None,
                    "best_accuracy": None,
                    "time_to_converge_s": self._time_to_converge,
                    "target_acc": self._target_acc,
                    "total_comm_bytes": 0,
                    "total_train_time_s": 0.0,
                    "total_comm_time_s": 0.0,
                    "avg_samples_per_sec": 0.0,
                }
            best = max(r.accuracy for r in rounds)
            return {
                "rounds_completed": len(rounds),
                "final_accuracy": rounds[-1].accuracy,
                "best_accuracy": best,
                "time_to_converge_s": self._time_to_converge,
                "target_acc": self._target_acc,
                "total_comm_bytes": sum(r.comm_bytes for r in rounds),
                "total_train_time_s": sum(r.train_time_s for r in rounds),
                "total_comm_time_s": sum(r.comm_time_s for r in rounds),
                "avg_samples_per_sec": (
                    sum(r.samples_per_sec for r in rounds) / len(rounds)
                ),
            }

    def full_blob(self) -> Dict:
        """Exact on-disk structure (after reload)."""
        self.reload_from_disk()
        with self._lock:
            return {
                "rounds": [asdict(r) for r in self._rounds],
                "time_to_converge": self._time_to_converge,
                "target_acc": self._target_acc,
            }

    def render_chart_png(self) -> bytes:
        """No matplotlib; placeholder PNG only."""
        return _TINY_PNG


_GLOBAL_STORE: Optional[MetricsStore] = None


def get_store() -> MetricsStore:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = MetricsStore()
    return _GLOBAL_STORE
