"""
server_tf.py  —  Flower FedAvg server with TensorFlow (for TFLite mobile clients)
Run this on your LAPTOP before starting any clients.

Usage:
    python server_tf.py --rounds 20 --clients 5 --variant small

The server:
  1. Broadcasts the initial global model to all clients
  2. Waits for client updates each round
  3. Aggregates with FedAvg
  4. Evaluates on global test set
  5. Saves round_log.json (accuracy, loss, upload KB per round)
"""

import argparse
import json
import time
import numpy as np
import tensorflow as tf
import flwr as fl
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict
from collections import OrderedDict

from tf_model_mobile import get_model, NUM_CLASSES
from data_utils_tf import get_client_loaders


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions for TensorFlow
# ══════════════════════════════════════════════════════════════════════════════

def get_parameters(model: tf.keras.Model) -> List[np.ndarray]:
    """Get model weights as numpy arrays."""
    return [w.numpy() for w in model.trainable_weights]


def set_parameters(model: tf.keras.Model, parameters: List[np.ndarray]):
    """Set model weights from numpy arrays."""
    for weight, param in zip(model.trainable_weights, parameters):
        weight.assign(param)


def evaluate_global(model: tf.keras.Model, test_loader, device="cpu"):
    """Evaluate model on test set."""
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    loss, accuracy = model.evaluate(test_loader, verbose=0)
    return loss, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# Logging strategy with TensorFlow
# ══════════════════════════════════════════════════════════════════════════════

class LoggingFedAvg(FedAvg):
    """FedAvg + per-round logging to round_log.json."""

    def __init__(self, *args, variant="small", test_loader=None, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.test_loader = test_loader
        self.device = device
        self.round_log = []
        self.start_time = time.time()

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        # Collect metrics from clients
        if results:
            train_losses = [r.metrics.get("train_loss", 0.0) for _, r in results]
            train_times = [r.metrics.get("train_time", 0.0) for _, r in results]
            upload_kbs = [r.metrics.get("upload_kb", 0.0) for _, r in results]

            # Server-side global eval
            if aggregated[0] is not None and self.test_loader is not None:
                model = get_model(self.variant)
                params = parameters_to_ndarrays(aggregated[0])
                set_parameters(model, params)
                s_loss, s_acc = evaluate_global(model, self.test_loader, self.device)
            else:
                s_loss, s_acc = 0.0, 0.0

            entry = {
                "round": server_round,
                "elapsed_s": round(time.time() - self.start_time, 2),
                "num_clients": len(results),
                "avg_train_loss": float(np.mean(train_losses)),
                "avg_train_time_s": float(np.mean(train_times)),
                "total_upload_kb": float(np.sum(upload_kbs)),
                "server_loss": float(s_loss),
                "server_accuracy": float(s_acc),
            }
            self.round_log.append(entry)

            print(
                f"[Round {server_round:3d}]  "
                f"clients={len(results)}  "
                f"train_loss={entry['avg_train_loss']:.4f}  "
                f"server_acc={s_acc:.4f}  "
                f"upload={entry['total_upload_kb']:.1f} KB"
            )

            # Save after every round
            with open("round_log.json", "w") as f:
                json.dump(self.round_log, f, indent=2)

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)


# ══════════════════════════════════════════════════════════════════════════════
# Weighted-average metric aggregation
# ══════════════════════════════════════════════════════════════════════════════

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    agg = {}
    for n, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v * n / total
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Server (TensorFlow)")
    parser.add_argument("--rounds", type=int, default=20, help="Number of FL rounds")
    parser.add_argument("--clients", type=int, default=5, help="Min clients per round")
    parser.add_argument("--variant", type=str, default="small", 
                        choices=["large", "medium", "small"],
                        help="Model variant (small recommended for mobile)")
    parser.add_argument("--port", type=int, default=8080, help="gRPC port")
    parser.add_argument("--alpha", type=float, default=0.5, 
                        help="Dirichlet alpha for data split")
    args = parser.parse_args()

    print(f"[server] Starting Flower server (TensorFlow)")
    print(f"         model variant : {args.variant}")
    print(f"         FL rounds     : {args.rounds}")
    print(f"         min clients   : {args.clients}")
    print(f"         port          : {args.port}")
    print(f"         Dirichlet α   : {args.alpha}")
    print()

    # Load test set for global evaluation
    _, test_loader, _ = get_client_loaders(
        num_clients=args.clients, batch_size=256, alpha=args.alpha
    )
    device = "cuda" if tf.config.list_physical_devices('GPU') else "cpu"

    # Initial model parameters broadcast to clients
    init_model = get_model(args.variant)
    init_params = ndarrays_to_parameters(get_parameters(init_model))

    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=init_params,
        fit_metrics_aggregation_fn=weighted_average,
        variant=args.variant,
        test_loader=test_loader,
        device=device,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print("\n[server] Training complete. Round log saved to round_log.json")


if __name__ == "__main__":
    main()
