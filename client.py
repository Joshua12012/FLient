"""
client.py  —  Flower edge client
Works identically on:
  • Android phones via Termux
  • Simulated clients on a PC

Usage (phone / Termux):
    python client.py --server 192.168.1.100:8080 --client_id 0 --num_clients 10

Usage (simulated on PC — run multiple times with different client_id):
    python client.py --server 127.0.0.1:8080 --client_id 2 --num_clients 10 --simulate

Edge-realism features included:
  • Straggler delay simulation
  • Bandwidth-limited upload simulation
  • Per-round metrics: train_loss, train_time, upload_kb
"""

import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict

from model import get_model, NUM_CLASSES
from data_utils import get_single_client_loader


# ── parameter helpers ─────────────────────────────────────────────────────────

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# ── upload size estimator ─────────────────────────────────────────────────────

def estimate_upload_kb(model):
    """Approximates the size of model weights if sent over the wire (float32)."""
    total_bytes = sum(p.numel() * 4 for p in model.parameters())
    return total_bytes / 1024


# ── Flower client ─────────────────────────────────────────────────────────────

class FEMNISTClient(fl.client.NumPyClient):

    def __init__(self, client_id, num_clients, variant, alpha,
                 bandwidth_mbps, straggler, epochs, batch_size, simulate):
        self.client_id      = client_id
        self.variant        = variant
        self.bandwidth_mbps = bandwidth_mbps
        self.straggler      = straggler
        self.epochs         = epochs
        self.simulate       = simulate
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"

        # Load this client's shard of FEMNIST
        print(f"[client {client_id}] Loading FEMNIST shard…")
        self.train_loader, self.test_loader, _ = get_single_client_loader(
            client_id=client_id,
            num_clients=num_clients,
            batch_size=batch_size,
            alpha=alpha,
        )
        print(f"[client {client_id}] Shard size: {len(self.train_loader.dataset)} samples")

        self.model     = get_model(variant).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    # ── Flower API ──────────────────────────────────────────────────────────

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # 1. Sync global weights
        set_parameters(self.model, parameters)

        # 2. Simulate straggler delay (slow phones finish later)
        if self.straggler:
            delay = random.uniform(1.0, 5.0)
            print(f"[client {self.client_id}] Straggler delay: {delay:.1f}s")
            time.sleep(delay)

        # 3. Local training
        t0 = time.time()
        train_loss = self._train(self.epochs)
        train_time = time.time() - t0

        # 4. Simulate upload delay based on bandwidth
        upload_kb = estimate_upload_kb(self.model)
        if self.simulate:
            upload_delay = (upload_kb / 1024) / self.bandwidth_mbps
            time.sleep(upload_delay)

        print(
            f"[client {self.client_id}] "
            f"loss={train_loss:.4f}  "
            f"time={train_time:.1f}s  "
            f"upload={upload_kb:.1f} KB"
        )

        metrics = {
            "train_loss":  float(train_loss),
            "train_time":  float(train_time),
            "upload_kb":   float(upload_kb),
            "client_id":   float(self.client_id),
        }
        return get_parameters(self.model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = self._evaluate()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    # ── internal helpers ────────────────────────────────────────────────────

    def _train(self, epochs):
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        total_loss = 0.0
        total_samples = 0
        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out  = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss    += loss.item() * x.size(0)
                total_samples += x.size(0)
        return total_loss / max(total_samples, 1)

    def _evaluate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out   = self.model(x)
                loss  = self.criterion(out, y)
                total_loss += loss.item() * x.size(0)
                pred        = out.argmax(dim=1)
                correct    += pred.eq(y).sum().item()
                total      += x.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Flower FEMNIST Edge Client")
    parser.add_argument("--server",       type=str,   default="127.0.0.1:8080",
                        help="Server IP:port (use your laptop IP on WiFi)")
    parser.add_argument("--client_id",    type=int,   default=0,
                        help="Unique ID for this client (0, 1, 2, …)")
    parser.add_argument("--num_clients",  type=int,   default=5,
                        help="Total number of clients (must match server)")
    parser.add_argument("--variant",      type=str,   default="large",
                        choices=["large","medium","small"],
                        help="Model size (adaptive_serving.py picks this automatically)")
    parser.add_argument("--epochs",       type=int,   default=3,
                        help="Local epochs per round")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--alpha",        type=float, default=0.5,
                        help="Dirichlet alpha (must match server)")
    parser.add_argument("--bandwidth",    type=float, default=10.0,
                        help="Simulated bandwidth in Mbps (for upload delay)")
    parser.add_argument("--straggler",    action="store_true",
                        help="Enable random straggler delay")
    parser.add_argument("--simulate",     action="store_true",
                        help="Enable simulated upload delay (for PC clients)")
    args = parser.parse_args()

    print(f"[client {args.client_id}] Connecting to server at {args.server}")
    print(f"  variant={args.variant}  epochs={args.epochs}  "
          f"batch_size={args.batch_size}  alpha={args.alpha}")

    client = FEMNISTClient(
        client_id     = args.client_id,
        num_clients   = args.num_clients,
        variant       = args.variant,
        alpha         = args.alpha,
        bandwidth_mbps= args.bandwidth,
        straggler     = args.straggler,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        simulate      = args.simulate,
    )

    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )


if __name__ == "__main__":
    main()
