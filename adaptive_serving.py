"""
adaptive_serving.py
-------------------
Adaptive Inference Deployment for Edge Devices.

Problem:
  Edge devices have wildly different capabilities (CPU speed, RAM, battery).
  Serving the same large model to every device is wasteful / infeasible.

Solution – Adaptive Serving:
  1. DEVICE PROFILER : measures available compute and memory.
  2. MODEL SELECTOR  : picks the best model variant for that device tier.
  3. LATENCY BENCHMARKER : runs a quick forward-pass timing test.
  4. INFERENCE ENGINE : wraps the selected model with async-ready inference.

Device Tiers (simulated)
  TIER_HIGH   → full EdgeCNN_Full   (laptops, powerful phones)
  TIER_MEDIUM → EdgeCNN_Medium      (mid-range phones, Raspberry Pi 4)
  TIER_LOW    → EdgeCNN_Small       (microcontrollers, old phones)

Knowledge Distillation hint (architecture note):
  In production the MEDIUM/SMALL models would be trained via knowledge
  distillation from the LARGE teacher model.  Here we demonstrate the
  serving selection logic; the distillation training loop is described
  in the comments.
"""

import time
import platform
import psutil
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from model import get_model, count_parameters


# ---------------------------------------------------------------------------
# Device Profile
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    cpu_cores:      int
    ram_mb:         float
    cpu_freq_mhz:   float
    platform:       str
    tier:           str = field(init=False)   # assigned by profiler

    def __post_init__(self):
        self.tier = _assign_tier(self.cpu_cores, self.ram_mb, self.cpu_freq_mhz)

    def summary(self) -> str:
        return (
            f"Device Profile\n"
            f"  Platform   : {self.platform}\n"
            f"  CPU Cores  : {self.cpu_cores}\n"
            f"  RAM        : {self.ram_mb:.0f} MB\n"
            f"  CPU Freq   : {self.cpu_freq_mhz:.0f} MHz\n"
            f"  Tier       : {self.tier.upper()}\n"
        )


def _assign_tier(cores: int, ram_mb: float, freq_mhz: float) -> str:
    score = cores * 0.4 + (ram_mb / 1024) * 0.4 + (freq_mhz / 1000) * 0.2
    if score >= 3.0:
        return "high"
    if score >= 1.5:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Device Profiler
# ---------------------------------------------------------------------------

class DeviceProfiler:
    """Inspect the current host to determine its hardware tier."""

    @staticmethod
    def profile() -> DeviceProfile:
        cores    = psutil.cpu_count(logical=True) or 2
        ram_mb   = psutil.virtual_memory().total / (1024 ** 2)
        freq_info = psutil.cpu_freq()
        freq_mhz  = freq_info.current if freq_info else 1000.0
        plat      = f"{platform.system()} {platform.machine()}"

        return DeviceProfile(
            cpu_cores    = cores,
            ram_mb       = ram_mb,
            cpu_freq_mhz = freq_mhz,
            platform     = plat,
        )

    @staticmethod
    def simulate(tier: str) -> DeviceProfile:
        """Simulate a specific device tier for testing."""
        profiles = {
            "high":   DeviceProfile(cpu_cores=8,  ram_mb=8192, cpu_freq_mhz=3200, platform="Simulated/x86_64"),
            "medium": DeviceProfile(cpu_cores=4,  ram_mb=2048, cpu_freq_mhz=1800, platform="Simulated/ARM64"),
            "low":    DeviceProfile(cpu_cores=1,  ram_mb=256,  cpu_freq_mhz=600,  platform="Simulated/ARM32"),
        }
        return profiles[tier]


# ---------------------------------------------------------------------------
# Model Selector
# ---------------------------------------------------------------------------

TIER_TO_MODEL: Dict[str, str] = {
    "high":   "large",
    "medium": "medium",
    "low":    "small",
}


class ModelSelector:
    """Selects the optimal model variant based on device tier."""

    @staticmethod
    def select(profile: DeviceProfile) -> str:
        return TIER_TO_MODEL[profile.tier]

    @staticmethod
    def explain(profile: DeviceProfile) -> str:
        variant = TIER_TO_MODEL[profile.tier]
        model   = get_model(variant)
        params  = count_parameters(model)
        return (
            f"Selected Model Variant: {variant.upper()}\n"
            f"  Parameters : {params:,}\n"
            f"  Rationale  : Device tier={profile.tier}, "
            f"RAM={profile.ram_mb:.0f}MB, Cores={profile.cpu_cores}\n"
        )


# ---------------------------------------------------------------------------
# Latency Benchmarker
# ---------------------------------------------------------------------------

class LatencyBenchmarker:
    """
    Measures forward-pass latency and estimates throughput.
    Runs N warm-up passes then K timed passes.
    """

    def __init__(self, warmup: int = 3, runs: int = 10, batch_size: int = 1):
        self.warmup     = warmup
        self.runs       = runs
        self.batch_size = batch_size

    def benchmark(self, model: torch.nn.Module, device: str = "cpu") -> Dict:
        model = model.to(device)
        model.eval()
        dummy = torch.zeros(self.batch_size, 1, 28, 28).to(device)

        # Warm-up
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = model(dummy)

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(self.runs):
                t0 = time.perf_counter()
                _  = model(dummy)
                times.append(time.perf_counter() - t0)

        latency_ms = np.array(times) * 1000
        return {
            "mean_latency_ms":   round(float(latency_ms.mean()), 3),
            "p95_latency_ms":    round(float(np.percentile(latency_ms, 95)), 3),
            "p99_latency_ms":    round(float(np.percentile(latency_ms, 99)), 3),
            "throughput_fps":    round(1000.0 / float(latency_ms.mean()), 1),
            "batch_size":        self.batch_size,
        }


# ---------------------------------------------------------------------------
# Adaptive Inference Engine
# ---------------------------------------------------------------------------

class AdaptiveInferenceEngine:
    """
    High-level inference engine that:
      1. Profiles the device.
      2. Selects the appropriate model.
      3. Loads trained weights (if provided).
      4. Runs benchmarks.
      5. Exposes a simple predict() method.
    """

    def __init__(
        self,
        weights: Optional[list] = None,
        force_tier: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = device

        # 1. Profile
        self.profile  = (DeviceProfiler.simulate(force_tier)
                         if force_tier else DeviceProfiler.profile())

        # 2. Select model
        self.variant  = ModelSelector.select(self.profile)
        self.model    = get_model(self.variant).to(device)

        # 3. Load weights
        if weights is not None:
            keys  = list(self.model.state_dict().keys())
            state = {k: torch.tensor(w)
                     for k, w in zip(keys, weights)}
            self.model.load_state_dict(state, strict=True)
            print(f"[Engine] Loaded federated weights into {self.variant} model.")

        self.model.eval()

        # 4. Benchmark
        bench = LatencyBenchmarker()
        self.bench_results = bench.benchmark(self.model, device)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (N, 1, 28, 28)

        Returns
        -------
        probs  : softmax probabilities (N, 10)
        labels : predicted class indices (N,)
        """
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=-1)
            labels = probs.argmax(dim=-1)
        return probs, labels

    def report(self) -> str:
        b = self.bench_results
        lines = [
            "\n" + "─" * 50,
            " ADAPTIVE INFERENCE ENGINE REPORT",
            "─" * 50,
            self.profile.summary(),
            ModelSelector.explain(self.profile),
            "Latency Benchmark",
            f"  Mean latency : {b['mean_latency_ms']} ms",
            f"  P95  latency : {b['p95_latency_ms']}  ms",
            f"  P99  latency : {b['p99_latency_ms']}  ms",
            f"  Throughput   : {b['throughput_fps']} FPS",
            "─" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for tier in ("high", "medium", "low"):
        engine = AdaptiveInferenceEngine(force_tier=tier)
        print(engine.report())

        dummy = torch.zeros(4, 1, 28, 28)
        probs, preds = engine.predict(dummy)
        print(f"  Sample predictions: {preds.tolist()}\n")
