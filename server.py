"""
server.py  —  Flower FedAvg server with per-round logging
Run this on your LAPTOP before starting any clients.

Usage:
    python server.py --rounds 20 --clients 5 --variant large

The server:
  1. Broadcasts the initial global model to all clients
  2. Waits for client updates each round
  3. Aggregates with FedAvg
  4. Evaluates on global test set
  5. Saves round_log.json  (accuracy, loss, upload KB per round)
"""

import argparse
import json
import time
import numpy as np
import tensorflow as tf
import flwr as fl
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Optional, Dict, Union
from collections import OrderedDict

from tf_model import get_tf_model, NUM_CLASSES


# ── helpers ───────────────────────────────────────────────────────────────────

def get_parameters(model):
    """Extract model weights as NumPy arrays for Flower."""
    return [w.numpy() for w in model.trainable_weights]


def set_parameters(model, parameters):
    """Set model weights from NumPy arrays from Flower."""
    model.trainable_weights[0].assign(parameters[0])
    for i, w in enumerate(model.trainable_weights[1:]):
        w.assign(parameters[i+1])


def evaluate_global(model, test_x, test_y):
    """Evaluate TensorFlow model on test data."""
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    return loss, accuracy


# ── logging strategy ──────────────────────────────────────────────────────────

class LoggingFedAvg(FedAvg):
    """FedAvg + per-round logging to round_log.json."""

    def __init__(self, *args, variant="large", test_x=None, test_y=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant     = variant
        self.test_x      = test_x
        self.test_y      = test_y
        self.round_log   = []
        self.start_time  = time.time()

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        # Collect metrics from clients
        if results:
            train_losses = [r.metrics.get("train_loss", 0.0) for _, r in results]
            train_times  = [r.metrics.get("train_time", 0.0) for _, r in results]
            upload_kbs   = [r.metrics.get("upload_kb",  0.0) for _, r in results]

            # Server-side global eval
            if aggregated[0] is not None and self.test_x is not None:
                model = get_tf_model(self.variant)
                params = parameters_to_ndarrays(aggregated[0])
                set_parameters(model, params)
                s_loss, s_acc = evaluate_global(model, self.test_x, self.test_y)
            else:
                s_loss, s_acc = 0.0, 0.0

            entry = {
                "round":            server_round,
                "elapsed_s":        round(time.time() - self.start_time, 2),
                "num_clients":      len(results),
                "avg_train_loss":   float(np.mean(train_losses)),
                "avg_train_time_s": float(np.mean(train_times)),
                "total_upload_kb":  float(np.sum(upload_kbs)),
                "server_loss":      float(s_loss),
                "server_accuracy":  float(s_acc),
            }
            self.round_log.append(entry)

            print(
                f"[Round {server_round:3d}]  "
                f"clients={len(results)}  "
                f"train_loss={entry['avg_train_loss']:.4f}  "
                f"server_acc={s_acc:.4f}  "
                f"upload={entry['total_upload_kb']:.1f} KB"
            )

            # Save after every round so it's never lost
            with open("round_log.json", "w") as f:
                json.dump(self.round_log, f, indent=2)

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)


# ── weighted-average metric aggregation helpers ───────────────────────────────

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    agg   = {}
    for n, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v * n / total
    return agg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Server")
    parser.add_argument("--rounds",   type=int, default=20,    help="Number of FL rounds")
    parser.add_argument("--clients",  type=int, default=5,     help="Min clients per round")
    parser.add_argument("--variant",  type=str, default="large", choices=["large","medium","small"])
    parser.add_argument("--port",     type=int, default=8080,  help="gRPC port")
    parser.add_argument("--alpha",    type=float, default=0.5, help="Dirichlet alpha for data split")
    args = parser.parse_args()

    print(f"[server] Starting Flower server")
    print(f"         model variant : {args.variant}")
    print(f"         FL rounds     : {args.rounds}")
    print(f"         min clients   : {args.clients}")
    print(f"         port          : {args.port}")
    print(f"         Dirichlet α   : {args.alpha}")
    print()

    # Generate synthetic test data for global evaluation
    print(f"[server] Generating synthetic test data...")
    test_x = np.random.randn(1000, 28, 28, 1).astype(np.float32)
    test_y = np.random.randint(0, NUM_CLASSES, 1000)
    print(f"[server] Test data ready: {len(test_x)} samples")

    # Initial model parameters broadcast to clients
    init_model  = get_tf_model(args.variant)
    init_params = ndarrays_to_parameters(get_parameters(init_model))

    strategy = LoggingFedAvg(
        fraction_fit=1.0,                      # use ALL available clients each round
        fraction_evaluate=0.0,                  # server does its own eval
        min_fit_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=init_params,
        fit_metrics_aggregation_fn=weighted_average,
        variant=args.variant,
        test_x=test_x,
        test_y=test_y,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print("\n[server] Training complete. Round log saved to round_log.json")


if __name__ == "__main__":
    main()
