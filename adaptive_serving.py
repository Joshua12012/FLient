"""
adaptive_serving.py  —  Device profiler + adaptive model tier selection

Run this on EACH device (phone or PC) before launching client.py.
It inspects the hardware, picks the right model variant, and prints the
client.py command to run.

Tiers:
    high   → large model  (RAM ≥ 4 GB, cores ≥ 4)
    medium → medium model (RAM 2–4 GB, or cores 2–3)
    low    → small model  (RAM < 2 GB, or single core)
"""

import platform
import subprocess
import sys
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ── hardware profiler ─────────────────────────────────────────────────────────

def profile_device():
    """Returns a dict of hardware metrics. Works on Android/Termux and PC."""
    info = {
        "platform":   platform.system(),
        "machine":    platform.machine(),
        "python":     platform.python_version(),
        "cpu_cores":  os.cpu_count() or 1,
        "ram_gb":     0.0,
        "cpu_freq_mhz": 0.0,
    }

    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        info["ram_gb"] = mem.total / (1024 ** 3)
        freq = psutil.cpu_freq()
        if freq:
            info["cpu_freq_mhz"] = freq.current
    else:
        # Fallback: read /proc/meminfo (works on Android/Termux)
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = kb / (1024 ** 2)
                        break
        except Exception:
            info["ram_gb"] = 1.0   # safe default

        # Fallback: read CPU freq from /sys
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") as f:
                info["cpu_freq_mhz"] = int(f.read().strip()) / 1000
        except Exception:
            info["cpu_freq_mhz"] = 1000.0

    return info


def assign_tier(info):
    """
    Assigns a tier based on RAM and CPU cores.
    Returns ('high'|'medium'|'low', model_variant, reason)
    """
    ram   = info["ram_gb"]
    cores = info["cpu_cores"]

    if ram >= 4.0 and cores >= 4:
        return "high",   "large",  f"RAM={ram:.1f}GB cores={cores} → full model"
    elif ram >= 2.0 or cores >= 2:
        return "medium", "medium", f"RAM={ram:.1f}GB cores={cores} → medium model"
    else:
        return "low",    "small",  f"RAM={ram:.1f}GB cores={cores} → small model"


# ── model loader (used when doing inference, not training) ────────────────────

def load_model_for_tier(variant):
    """Loads the appropriate model and returns it in eval mode."""
    import torch
    from model import get_model
    model = get_model(variant)
    model.eval()
    return model


# ── split inference demo ──────────────────────────────────────────────────────

def demo_split_inference():
    """
    Demonstrates hybrid parallelism:
    The device part runs locally, then the 128-d feature vector
    would be sent to the server for the final classification.
    (Here we run both parts locally to demo the concept.)
    """
    import torch
    from model import get_split_model, FEATURE_DIM

    print("\n[split inference demo]")
    device_part, server_part = get_split_model()
    device_part.eval()
    server_part.eval()

    # Simulate a single input image
    x = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        feature_vec = device_part(x)
        print(f"  Device part output : {feature_vec.shape}  "
              f"(this is what gets sent over WiFi)")
        print(f"  Feature vector size: {feature_vec.numel() * 4} bytes  "
              f"vs raw image: {x.numel() * 4} bytes")
        compression = x.numel() / feature_vec.numel()
        print(f"  Data reduction     : {compression:.1f}×")

        logits = server_part(feature_vec)
        pred   = logits.argmax(dim=1).item()
        print(f"  Server part output : {logits.shape}  predicted class: {pred}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive model tier selector")
    parser.add_argument("--server",      type=str, default="192.168.1.100:8080")
    parser.add_argument("--client_id",   type=int, default=0)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--demo_split",  action="store_true",
                        help="Run the split inference demo")
    args = parser.parse_args()

    print("=" * 55)
    print("  Adaptive serving — device profiler")
    print("=" * 55)

    info = profile_device()
    print(f"\n  Platform   : {info['platform']} {info['machine']}")
    print(f"  Python     : {info['python']}")
    print(f"  CPU cores  : {info['cpu_cores']}")
    print(f"  RAM        : {info['ram_gb']:.2f} GB")
    print(f"  CPU freq   : {info['cpu_freq_mhz']:.0f} MHz")

    tier, variant, reason = assign_tier(info)
    print(f"\n  Assigned tier  : {tier.upper()}")
    print(f"  Model variant  : {variant}")
    print(f"  Reason         : {reason}")

    print("\n" + "=" * 55)
    print("  Run this command to start your Flower client:")
    print("=" * 55)
    cmd = (
        f"python client.py "
        f"--server {args.server} "
        f"--client_id {args.client_id} "
        f"--num_clients {args.num_clients} "
        f"--variant {variant} "
        f"--epochs {args.epochs} "
        f"--alpha {args.alpha}"
    )
    print(f"\n  {cmd}\n")

    # Optionally run split inference demo
    if args.demo_split:
        demo_split_inference()

    return variant, cmd


if __name__ == "__main__":
    main()
