"""
comm_analysis.py  —  Plots round-by-round metrics from round_log.json

Run after training finishes:
    python comm_analysis.py

Generates:
    fl_results.png  — 4-panel chart: accuracy, loss, upload KB, training time
"""

import json
import sys
import os

try:
    import matplotlib
    matplotlib.use("Agg")            # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_log(path="round_log.json"):
    if not os.path.exists(path):
        print(f"[comm_analysis] '{path}' not found. Run training first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def print_summary(log):
    print("\n" + "=" * 60)
    print("  Communication Round Analysis")
    print("=" * 60)
    print(f"  Total rounds     : {len(log)}")
    print(f"  Total upload     : {sum(r['total_upload_kb'] for r in log):.1f} KB")
    print(f"  Avg upload/round : {sum(r['total_upload_kb'] for r in log) / len(log):.1f} KB")
    print(f"  Final accuracy   : {log[-1]['server_accuracy']:.4f}")
    print(f"  Best accuracy    : {max(r['server_accuracy'] for r in log):.4f}  "
          f"(round {max(log, key=lambda r: r['server_accuracy'])['round']})")
    print(f"  Final loss       : {log[-1]['server_loss']:.4f}")
    print(f"  Total wall time  : {log[-1]['elapsed_s']:.1f} s")
    print("=" * 60)


def plot_results(log, out_path="fl_results.png"):
    if not MATPLOTLIB_AVAILABLE:
        print("[comm_analysis] matplotlib not installed. Skipping plot.")
        print("  pip install matplotlib  (or  pip install matplotlib --break-system-packages)")
        return

    rounds    = [r["round"]            for r in log]
    accuracy  = [r["server_accuracy"]  for r in log]
    s_loss    = [r["server_loss"]       for r in log]
    t_loss    = [r["avg_train_loss"]   for r in log]
    upload_kb = [r["total_upload_kb"]  for r in log]
    train_time= [r["avg_train_time_s"] for r in log]
    cum_upload= []
    running   = 0
    for kb in upload_kb:
        running += kb
        cum_upload.append(running)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Federated Learning — Round Analysis (FEMNIST)", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1 — accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rounds, accuracy, "o-", color="#185FA5", linewidth=2, markersize=4)
    ax1.set_title("Global accuracy per round")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Panel 2 — server + train loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rounds, s_loss, "o-", color="#993C1D", linewidth=2, markersize=4, label="server")
    ax2.plot(rounds, t_loss, "s--", color="#BA7517", linewidth=1.5, markersize=3, label="avg train")
    ax2.set_title("Loss per round")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3 — upload KB per round
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(rounds, upload_kb, color="#1D9E75", alpha=0.7)
    ax3.set_title("Upload per round (all clients)")
    ax3.set_xlabel("Round")
    ax3.set_ylabel("KB")
    ax3.grid(axis="y", alpha=0.3)

    # Panel 4 — cumulative upload
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.fill_between(rounds, cum_upload, alpha=0.3, color="#185FA5")
    ax4.plot(rounds, cum_upload, "-", color="#185FA5", linewidth=2)
    ax4.set_title("Cumulative upload (KB)")
    ax4.set_xlabel("Round")
    ax4.set_ylabel("KB")
    ax4.grid(alpha=0.3)

    # Panel 5 — avg training time per round
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(rounds, train_time, "o-", color="#7F77DD", linewidth=2, markersize=4)
    ax5.set_title("Avg client training time")
    ax5.set_xlabel("Round")
    ax5.set_ylabel("Seconds")
    ax5.grid(alpha=0.3)

    # Panel 6 — accuracy vs upload cost
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(cum_upload, accuracy, c=rounds, cmap="viridis", s=40, zorder=3)
    ax6.set_title("Accuracy vs cumulative upload")
    ax6.set_xlabel("Cumulative upload (KB)")
    ax6.set_ylabel("Accuracy")
    ax6.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=min(rounds), vmax=max(rounds)))
    plt.colorbar(sm, ax=ax6, label="Round", pad=0.02)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[comm_analysis] Plot saved → {out_path}")


def main():
    log = load_log()
    print_summary(log)
    plot_results(log)


if __name__ == "__main__":
    main()
