"""
fl_runner.py  —  Full end-to-end demo on a SINGLE PC (no phones needed)
                 Simulates N clients in separate processes using multiprocessing.

Use this to test everything locally before involving phones.
Then switch to server.py + client.py across devices for the real thing.

Usage:
    python fl_runner.py --rounds 10 --clients 5 --variant large
"""

import argparse
import multiprocessing
import subprocess
import sys
import time
import os

from comm_analysis import load_log, print_summary, plot_results


def run_server(rounds, clients, variant, alpha, port):
    """Launched in a separate process."""
    cmd = [
        sys.executable, "server.py",
        "--rounds",  str(rounds),
        "--clients", str(clients),
        "--variant", variant,
        "--alpha",   str(alpha),
        "--port",    str(port),
    ]
    subprocess.run(cmd)


def run_client(client_id, num_clients, variant, alpha, epochs, batch_size, port):
    """Launched in a separate process per simulated client."""
    cmd = [
        sys.executable, "client.py",
        "--server",      f"127.0.0.1:{port}",
        "--client_id",   str(client_id),
        "--num_clients", str(num_clients),
        "--variant",     variant,
        "--alpha",       str(alpha),
        "--epochs",      str(epochs),
        "--batch_size",  str(batch_size),
        "--simulate",                        # enables upload delay simulation
        "--straggler",                       # enables random straggler delay
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="FL Runner — end-to-end local simulation")
    parser.add_argument("--rounds",     type=int,   default=10)
    parser.add_argument("--clients",    type=int,   default=5)
    parser.add_argument("--variant",    type=str,   default="large",
                        choices=["large","medium","small"])
    parser.add_argument("--alpha",      type=float, default=0.5)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--port",       type=int,   default=8080)
    args = parser.parse_args()

    print("=" * 60)
    print("  FL Runner — local simulation")
    print("=" * 60)
    print(f"  Rounds     : {args.rounds}")
    print(f"  Clients    : {args.clients}")
    print(f"  Variant    : {args.variant}")
    print(f"  Alpha (α)  : {args.alpha}")
    print(f"  Epochs/rnd : {args.epochs}")
    print(f"  Dataset    : FEMNIST (62 classes)")
    print()

    # Launch server
    server_proc = multiprocessing.Process(
        target=run_server,
        args=(args.rounds, args.clients, args.variant, args.alpha, args.port)
    )
    server_proc.start()
    print("[runner] Server started, waiting 3s for it to be ready…")
    time.sleep(3)

    # Launch all clients
    client_procs = []
    for cid in range(args.clients):
        p = multiprocessing.Process(
            target=run_client,
            args=(cid, args.clients, args.variant, args.alpha,
                  args.epochs, args.batch_size, args.port)
        )
        p.start()
        client_procs.append(p)
        time.sleep(0.5)   # stagger launches slightly

    print(f"[runner] {args.clients} clients started")

    # Wait for everyone to finish
    for p in client_procs:
        p.join()
    server_proc.join()

    print("\n[runner] All processes finished.")

    # Analysis
    print("\n[runner] Running communication analysis…")
    log = load_log("round_log.json")
    print_summary(log)
    plot_results(log, "fl_results.png")
    print("\n[runner] Done. Check fl_results.png for charts.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
