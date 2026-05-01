"""
run_simulation.py  –  in-process FedAvg (no Ray)
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

from data_utils          import get_client_loaders, load_mnist, dirichlet_partition, partition_stats
from model               import get_model, count_parameters, EdgeCNN_DevicePart, EdgeCNN_ServerPart
from communication_analysis import CommunicationAnalysis
from adaptive_serving    import AdaptiveInferenceEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds",   type=int,   default=10)
    p.add_argument("--clients",  type=int,   default=5)
    p.add_argument("--alpha",    type=float, default=0.5)
    p.add_argument("--epochs",   type=int,   default=3)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--variant",  type=str,   default="large",
                   choices=["large","medium","small"])
    p.add_argument("--device",   type=str,   default="cpu")
    return p.parse_args()

# ── Weight helpers ──────────────────────────────────────────────────────────
def get_weights(model):
    return [p.data.clone() for p in model.parameters()]

def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data.copy_(w)

def fedavg(weight_list, counts):
    total = sum(counts)
    avg = [torch.zeros_like(w) for w in weight_list[0]]
    for ws, n in zip(weight_list, counts):
        for i, w in enumerate(ws):
            avg[i] += w * (n / total)
    return avg

# ── Training / eval ─────────────────────────────────────────────────────────
def local_train(model, loader, epochs, lr, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    tloss, nb = 0.0, 0
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(X), y)
            loss.backward(); opt.step()
            tloss += loss.item(); nb += 1
    return tloss / max(nb, 1)

def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tloss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            tloss += crit(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)
    return tloss / len(loader), correct / total

# ── Printers ─────────────────────────────────────────────────────────────────
def print_partition_stats(train_ds, partitions):
    stats = partition_stats(train_ds, partitions)
    print("\n" + "="*76)
    print("  NON-IID DATA DISTRIBUTION ACROSS EDGE CLIENTS  (Dirichlet alpha)")
    print("="*76)
    print(f"{'Client':<12}" + "".join(f"{'C'+str(c):>6}" for c in range(10)) + f"{'Total':>8}")
    print("-"*76)
    for i, s in enumerate(stats):
        total = sum(s.values())
        print(f"Client {i:<4}" + "".join(f"{s.get(c,0):>6}" for c in range(10)) + f"{total:>8}")
    print("="*76 + "\n")

def demo_hybrid_parallelism(device):
    print("\n" + "="*62)
    print("  HYBRID PARALLELISM DEMO")
    print("  Data  Parallelism  => 5 clients train on private local shards")
    print("  Model Parallelism  => EdgeCNN split into:")
    print("    [DEVICE PART]  conv1->pool->conv2->pool->fc => 128-d feature")
    print("    [SERVER PART]  fc1->fc2 => logits (runs on server)")
    print("    Only the 128-d vector crosses the network")
    print("="*62)
    dev = EdgeCNN_DevicePart().to(device)
    srv = EdgeCNN_ServerPart().to(device)
    dummy = torch.zeros(4, 1, 28, 28).to(device)
    with torch.no_grad():
        feat = dev(dummy)
        out  = srv(feat)
    orig_B = dummy.numel() * 4
    feat_B = feat.numel() * 4
    print(f"\n  Input  shape : {list(dummy.shape)}")
    print(f"  Feature shape: {list(feat.shape)}  <- only this is transmitted")
    print(f"  Output shape : {list(out.shape)}")
    print(f"  Device-part params : {count_parameters(dev):,}")
    print(f"  Server-part params : {count_parameters(srv):,}")
    print(f"  Bandwidth saved    : {orig_B}B -> {feat_B}B  ({(1-feat_B/orig_B)*100:.1f}% reduction)\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
    LOG_PATH = str(RESULTS_DIR / "round_log.json")

    print("\n" + "#"*62)
    print("  FEDERATED & HYBRID DISTRIBUTED LEARNING - EDGE DEVICES")
    print("#"*62)
    print(f"  Rounds={args.rounds}  Clients={args.clients}  alpha={args.alpha}")
    print(f"  LocalEpochs={args.epochs}  Variant={args.variant}  Device={args.device}")
    print("#"*62 + "\n")

    dev = args.device

    # 1. Data
    print("[1/5] Preparing non-IID data partitions ...")
    train_loaders, test_loader = get_client_loaders(
        num_clients=args.clients, alpha=args.alpha, batch_size=32)
    train_ds   = load_mnist(train=True)
    partitions = dirichlet_partition(train_ds, args.clients, args.alpha)
    print_partition_stats(train_ds, partitions)

    # 2. Hybrid Parallelism
    print("[2/5] Demonstrating hybrid parallelism ...")
    demo_hybrid_parallelism(dev)

    # 3. FL Simulation
    print("[3/5] Running FedAvg simulation ...\n")
    global_model   = get_model(args.variant).to(dev)
    global_weights = get_weights(global_model)
    round_log = []
    n_sel = max(1, int(args.clients * args.fraction))
    t0 = time.perf_counter()

    for rnd in range(1, args.rounds + 1):
        rnd_t = time.perf_counter()
        selected = np.random.choice(args.clients, n_sel, replace=False)
        all_w, counts, losses, upload_kbs = [], [], [], []

        for cid in selected:
            lm = get_model(args.variant).to(dev)
            set_weights(lm, global_weights)
            loss = local_train(lm, train_loaders[cid], args.epochs, 0.01, dev)
            uw = get_weights(lm)
            all_w.append(uw)
            counts.append(len(train_loaders[cid].dataset))
            losses.append(loss)
            upload_kbs.append(sum(w.numel()*4 for w in uw) / 1024)
            del lm

        global_weights = fedavg(all_w, counts)
        set_weights(global_model, global_weights)
        sloss, sacc = evaluate(global_model, test_loader, dev)

        entry = dict(
            round=rnd, elapsed_s=round(time.perf_counter()-t0, 3),
            num_clients=int(n_sel),
            avg_train_loss=round(float(np.mean(losses)), 5),
            server_loss=round(float(sloss), 5),
            server_accuracy=round(float(sacc), 5),
            total_upload_kb=round(float(sum(upload_kbs)), 2),
            total_samples=int(sum(counts)),
        )
        round_log.append(entry)
        print(f"  Rnd {rnd:>2}/{args.rounds}  clients={n_sel}  "
              f"upload={entry['total_upload_kb']:.0f}KB  "
              f"loss={sloss:.4f}  acc={sacc*100:.2f}%  "
              f"({time.perf_counter()-rnd_t:.1f}s)")

    with open(LOG_PATH,"w") as f: json.dump(round_log, f, indent=2)
    print(f"\n[OK] Round log saved -> {LOG_PATH}")

    # 4. Analysis
    print("\n[4/5] Running communication round analysis ...")
    CommunicationAnalysis(LOG_PATH, str(RESULTS_DIR)).run_all()

    # 5. Adaptive Serving
    print("[5/5] Adaptive inference deployment demo ...\n")
    for tier in ("high","medium","low"):
        eng = AdaptiveInferenceEngine(force_tier=tier, device=dev)
        print(eng.report())
        _, p = eng.predict(torch.zeros(1,1,28,28))
        print(f"  Sample prediction: class {p.item()}\n")

    print("\n" + "#"*62)
    print("  SIMULATION COMPLETE  -  results saved to ./results/")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"    * {f.name:<40} ({f.stat().st_size/1024:.1f} KB)")
    print("#"*62 + "\n")

if __name__ == "__main__":
    main()
