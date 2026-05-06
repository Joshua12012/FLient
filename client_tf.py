"""
client_tf.py  —  TensorFlow Flower edge client
Works with TensorFlow/TFLite for mobile federated learning.

Usage (phone / Termux):
    python client_tf.py --server 192.168.1.100:8080 --client_id 0 --num_clients 10

Usage (simulated on PC):
    python client_tf.py --server 127.0.0.1:8080 --client_id 2 --num_clients 10 --simulate
"""

import argparse
import time
import random
import numpy as np
import tensorflow as tf
import flwr as fl
from typing import Dict, List, Tuple

from tf_model_mobile import get_model, NUM_CLASSES
from data_utils_tf import get_single_client_loader


# ══════════════════════════════════════════════════════════════════════════════
# Parameter helpers for TensorFlow
# ══════════════════════════════════════════════════════════════════════════════

def get_parameters(model: tf.keras.Model) -> List[np.ndarray]:
    """Get model weights as numpy arrays."""
    return [w.numpy() for w in model.trainable_weights]


def set_parameters(model: tf.keras.Model, parameters: List[np.ndarray]):
    """Set model weights from numpy arrays."""
    for weight, param in zip(model.trainable_weights, parameters):
        weight.assign(param)


def estimate_upload_kb(model: tf.keras.Model) -> float:
    """Approximate the size of model weights if sent over the wire (float32)."""
    total_bytes = sum(w.numpy().nbytes for w in model.trainable_weights)
    return total_bytes / 1024


# ══════════════════════════════════════════════════════════════════════════════
# TensorFlow Flower Client
# ══════════════════════════════════════════════════════════════════════════════

class FEMNISTClient(fl.client.NumPyClient):
    def __init__(self, client_id, num_clients, variant, alpha,
                 bandwidth_mbps, straggler, epochs, batch_size, simulate):
        self.client_id = client_id
        self.variant = variant
        self.bandwidth_mbps = bandwidth_mbps
        self.straggler = straggler
        self.epochs = epochs
        self.simulate = simulate
        
        # Load this client's shard of FEMNIST
        print(f"[client {client_id}] Loading FEMNIST shard…")
        self.train_loader, self.test_loader, _ = get_single_client_loader(
            client_id=client_id,
            num_clients=num_clients,
            batch_size=batch_size,
            alpha=alpha,
        )
        print(f"[client {client_id}] Shard size: {len(self.train_loader.dataset)} samples")
        
        # Build TensorFlow model
        self.model = get_model(variant)
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        self.initial_params = get_parameters(self.model)
    
    def get_parameters(self, config):
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        # 1. Sync global weights
        set_parameters(self.model, parameters)
        
        # 2. Simulate straggler delay
        if self.straggler:
            delay = random.uniform(1.0, 5.0)
            print(f"[client {self.client_id}] Straggler delay: {delay:.1f}s")
            time.sleep(delay)
        
        # 3. Local training
        t0 = time.time()
        train_loss = self._train(self.epochs)
        train_time = time.time() - t0
        
        # 4. Simulate upload delay
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
            "train_loss": float(train_loss),
            "train_time": float(train_time),
            "upload_kb": float(upload_kb),
            "client_id": float(self.client_id),
        }
        return get_parameters(self.model), len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = self._evaluate()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}
    
    def _train(self, epochs):
        """Local training on client's data."""
        self.model.fit(
            self.train_loader,
            epochs=epochs,
            verbose=0
        )
        
        # Return final loss
        loss, _ = self.model.evaluate(self.train_loader, verbose=0)
        return loss
    
    def _evaluate(self):
        """Evaluate model on test data."""
        loss, accuracy = self.model.evaluate(self.test_loader, verbose=0)
        return loss, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Flower FEMNIST Edge Client (TensorFlow)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080",
                        help="Server IP:port (use your laptop IP on WiFi)")
    parser.add_argument("--client_id", type=int, default=0,
                        help="Unique ID for this client (0, 1, 2, …)")
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Total number of clients (must match server)")
    parser.add_argument("--variant", type=str, default="small",
                        choices=["large", "medium", "small"],
                        help="Model size (small recommended for mobile)")
    parser.add_argument("--split_config", type=str, default="full_local",
                        help="Model split configuration (full_local only for now)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha (must match server)")
    parser.add_argument("--bandwidth", type=float, default=10.0,
                        help="Simulated bandwidth in Mbps")
    parser.add_argument("--straggler", action="store_true",
                        help="Enable random straggler delay")
    parser.add_argument("--simulate", action="store_true",
                        help="Enable simulated upload delay")
    args = parser.parse_args()
    
    if args.split_config == "server_only" and args.variant == "large":
        print("[client] split_config=server_only detected, lowering device model to small")
        args.variant = "small"
    
    print(f"[client {args.client_id}] Connecting to server at {args.server}")
    print(f"  variant={args.variant}  split_config={args.split_config}  epochs={args.epochs}  "
          f"batch_size={args.batch_size}  alpha={args.alpha}")
    
    client = FEMNISTClient(
        client_id=args.client_id,
        num_clients=args.num_clients,
        variant=args.variant,
        alpha=args.alpha,
        bandwidth_mbps=args.bandwidth,
        straggler=args.straggler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        simulate=args.simulate,
    )
    
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )


if __name__ == "__main__":
    main()
