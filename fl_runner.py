"""
fl_runner.py
------------
Efficient in-process FedAvg simulation.
Uses a configurable batch cap per client per epoch so the
simulation completes in reasonable time regardless of dataset size.
"""
import json, time, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path

from model      import get_model, EdgeCNN_DevicePart, EdgeCNN_ServerPart, count_parameters
from data_utils import get_client_loaders, load_mnist, dirichlet_partition, partition_stats
from communication_analysis import CommunicationAnalysis
from adaptive_serving       import AdaptiveInferenceEngine

# ── Config ───────────────────────────────────────────────────────────────────
ROUNDS         = 10
CLIENTS        = 5
ALPHA          = 0.5   # Dirichlet non-IID parameter
EPOCHS         = 2     # local epochs per round
BATCHES_PER_EP = 30    # max batches per client per epoch (keeps runtime short)
BATCH_SIZE     = 64
VARIANT        = "large"   # large / medium / small
DEVICE         = "cpu"
RESULTS_DIR    = Path("results")
LOG_PATH       = str(RESULTS_DIR / "round_log.json")
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(exist_ok=True)


# ── Weight helpers ────────────────────────────────────────────────────────────
def gw(m):
    return [p.data.clone() for p in m.parameters()]

def sw(m, ws):
    for p, w in zip(m.parameters(), ws):
        p.data.copy_(w)

def fedavg(wlist, counts):
    total = sum(counts)
    avg   = [torch.zeros_like(w) for w in wlist[0]]
    for ws, n in zip(wlist, counts):
        for i, w in enumerate(ws):
            avg[i] += w * (n / total)
    return avg


# ── Local training ────────────────────────────────────────────────────────────
def local_train(model, loader, epochs, lr, max_batches):
    model.train()
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    last_loss = 0.0
    for _ in range(epochs):
        for b, (X, y) in enumerate(loader):
            if b >= max_batches:
                break
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            last_loss = loss.item()
    return last_loss


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tl, ok, tot = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            out  = model(X)
            tl  += crit(out, y).item()
            ok  += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
    return tl / len(loader), ok / tot


# ── Pretty printers ───────────────────────────────────────────────────────────
def print_partition_stats(train_ds, partitions):
    stats = partition_stats(train_ds, partitions)
    print("\n" + "=" * 76)
    print("  NON-IID DATA DISTRIBUTION  (Dirichlet α =", ALPHA, ")")
    print("=" * 76)
    print(f"{'Client':<12}" + "".join(f"{'C'+str(c):>6}" for c in range(10)) + f"{'Total':>8}")
    print("-" * 76)
    for i, s in enumerate(stats):
        total = sum(s.values())
        print(f"Client {i:<4}" +
              "".join(f"{s.get(c,0):>6}" for c in range(10)) +
              f"{total:>8}")
    print("=" * 76 + "\n")


def demo_hybrid_parallelism():
    print("\n" + "=" * 62)
    print("  HYBRID PARALLELISM DEMONSTRATION")
    print("  ─── Data Parallelism ──────────────────────────────────")
    print("  • Each of the 5 clients trains on its own local shard")
    print("  • Raw data NEVER leaves the device (data locality)")
    print("  • Only model weights are exchanged with the server")
    print("  ─── Model Parallelism (Split Inference) ───────────────")
    print("  • EdgeCNN is split into DEVICE PART + SERVER PART")
    print("  • [DEVICE] conv1→pool→conv2→pool→fc → 128-d feature vec")
    print("  • [SERVER] fc1(128→64) → fc2(64→10) → logits")
    print("  • Only the 128-d feature vector crosses the network")
    print("=" * 62)
    dev_part = EdgeCNN_DevicePart()
    srv_part = EdgeCNN_ServerPart()
    dummy = torch.zeros(4, 1, 28, 28)
    with torch.no_grad():
        feat   = dev_part(dummy)
        logits = srv_part(feat)
    orig_B = dummy.numel() * 4
    feat_B = feat.numel() * 4
    print(f"\n  Input  shape : {list(dummy.shape)}")
    print(f"  Feature shape: {list(feat.shape)}  ← only this is transmitted")
    print(f"  Output shape : {list(logits.shape)}")
    print(f"  Device-part params : {count_parameters(dev_part):,}")
    print(f"  Server-part params : {count_parameters(srv_part):,}")
    print(f"  Bandwidth saved    : {orig_B}B → {feat_B}B  "
          f"({(1-feat_B/orig_B)*100:.1f}% reduction)\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 62, flush=True)
    print("  FEDERATED & HYBRID DISTRIBUTED LEARNING – EDGE DEVICES")
    print("#" * 62)
    print(f"  Rounds={ROUNDS}  Clients={CLIENTS}  α={ALPHA}")
    print(f"  LocalEpochs={EPOCHS}  BatchCap={BATCHES_PER_EP}  Variant={VARIANT}")
    print("#" * 62 + "\n", flush=True)

    # ── 1. Data ──────────────────────────────────────────────
    print("[1/5] Preparing non-IID data partitions ...", flush=True)
    train_loaders, test_loader = get_client_loaders(
        num_clients=CLIENTS, alpha=ALPHA, batch_size=BATCH_SIZE)
    train_ds   = load_mnist(train=True)
    partitions = dirichlet_partition(train_ds, CLIENTS, ALPHA)
    print_partition_stats(train_ds, partitions)

    # ── 2. Hybrid Parallelism ─────────────────────────────────
    print("[2/5] Demonstrating hybrid parallelism ...", flush=True)
    demo_hybrid_parallelism()

    # ── 3. FL Simulation ──────────────────────────────────────
    print("[3/5] Running FedAvg Federated Learning ...\n", flush=True)
    global_model   = get_model(VARIANT)
    global_weights = gw(global_model)
    round_log      = []
    sim_t0         = time.perf_counter()

    for rnd in range(1, ROUNDS + 1):
        rnd_t = time.perf_counter()
        all_w, counts, losses, upload_kbs = [], [], [], []

        for cid in range(CLIENTS):
            lm = get_model(VARIANT)
            sw(lm, global_weights)
            loss = local_train(lm, train_loaders[cid],
                               EPOCHS, lr=0.01, max_batches=BATCHES_PER_EP)
            uw = gw(lm)
            all_w.append(uw)
            counts.append(len(train_loaders[cid].dataset))
            losses.append(loss)
            upload_kbs.append(sum(w.numel() * 4 for w in uw) / 1024)
            del lm

        global_weights = fedavg(all_w, counts)
        sw(global_model, global_weights)
        sl, sa = evaluate(global_model, test_loader)

        entry = dict(
            round           = rnd,
            elapsed_s       = round(time.perf_counter() - sim_t0, 3),
            num_clients     = CLIENTS,
            avg_train_loss  = round(float(np.mean(losses)), 5),
            server_loss     = round(float(sl), 5),
            server_accuracy = round(float(sa), 5),
            total_upload_kb = round(float(sum(upload_kbs)), 2),
            total_samples   = int(sum(counts)),
        )
        round_log.append(entry)
        print(f"  Rnd {rnd:>2}/{ROUNDS}  "
              f"clients={CLIENTS}  upload={entry['total_upload_kb']:.0f} KB  "
              f"loss={sl:.4f}  acc={sa*100:.2f}%  "
              f"({time.perf_counter()-rnd_t:.1f}s)", flush=True)

    with open(LOG_PATH, "w") as f:
        json.dump(round_log, f, indent=2)
    print(f"\n[✓] Round log saved → {LOG_PATH}", flush=True)

    # ── 4. Communication Analysis ─────────────────────────────
    print("\n[4/5] Running communication round analysis ...", flush=True)
    CommunicationAnalysis(LOG_PATH, str(RESULTS_DIR)).run_all()

    # ── 5. Adaptive Serving ───────────────────────────────────
    print("[5/5] Demonstrating adaptive inference deployment ...\n", flush=True)
    for tier in ("high", "medium", "low"):
        engine = AdaptiveInferenceEngine(force_tier=tier, device=DEVICE)
        print(engine.report())
        dummy  = torch.zeros(1, 1, 28, 28)
        _, pred = engine.predict(dummy)
        print(f"  Sample prediction: class {pred.item()}\n")

    print("\n" + "#" * 62)
    print("  SIMULATION COMPLETE  –  results saved to ./results/")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"    • {f.name:<42} ({f.stat().st_size/1024:.1f} KB)")
    print("#" * 62 + "\n", flush=True)


if __name__ == "__main__":
    main()
