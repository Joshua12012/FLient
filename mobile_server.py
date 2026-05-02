"""
mobile_server.py — Lightweight Flower Server using TensorFlow

Uses TensorFlow/Keras on server side to match mobile_client.py.
Both client and server use the same framework for compatibility.
"""

import argparse
import json
import time
import numpy as np
import tensorflow as tf
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict
import logging
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.utils.connection_utils import find_available_port, is_port_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(num_classes=10):
    """
    Create TensorFlow/Keras model matching mobile_client.py.
    Same architecture ensures weight compatibility.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_parameters(model):
    """Extract parameters as numpy arrays."""
    return [w.numpy() for w in model.trainable_weights]


def set_parameters(model, parameters):
    """Set parameters from numpy arrays."""
    weight_shapes = [w.shape for w in model.trainable_weights]
    weight_values = []
    idx = 0
    
    for shape in weight_shapes:
        size = np.prod(shape)
        weight_values.append(parameters[idx].reshape(shape))
        idx += 1
    
    model.set_weights(weight_values)


class MobileFedAvg(FedAvg):
    """FedAvg strategy with logging for mobile clients."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_log = []
        self.start_time = time.time()
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results from clients."""
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if results:
            # Collect metrics
            train_losses = [r.metrics.get("train_loss", 0.0) for _, r in results]
            train_times = [r.metrics.get("train_time", 0.0) for _, r in results]
            client_ids = [r.metrics.get("client_id", -1) for _, r in results]
            
            entry = {
                "round": server_round,
                "elapsed_s": round(time.time() - self.start_time, 2),
                "num_clients": len(results),
                "avg_train_loss": float(np.mean(train_losses)),
                "avg_train_time_s": float(np.mean(train_times)),
                "clients": [int(c) for c in client_ids],
            }
            self.round_log.append(entry)
            
            print(
                f"[Round {server_round:3d}] "
                f"clients={len(results)} "
                f"train_loss={entry['avg_train_loss']:.4f} "
                f"avg_time={entry['avg_train_time_s']:.1f}s"
            )
            
            # Save after every round
            with open("mobile_round_log.json", "w") as f:
                json.dump(self.round_log, f, indent=2)
        
        return aggregated


def main():
    parser = argparse.ArgumentParser(description="Mobile TensorFlow Flower Server")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--clients", type=int, default=3, help="Min clients per round")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of output classes")
    args = parser.parse_args()
    
    # Auto-find available port if default is in use
    port = args.port
    if not is_port_available(port):
        logger.warning(f"Port {port} is in use, searching for available port...")
        try:
            port = find_available_port(start_port=port + 1)
            logger.info(f"Using alternative port: {port}")
        except RuntimeError as e:
            logger.error(f"Could not find available port: {e}")
            raise
    
    print(f"[Mobile Server] Starting TensorFlow Flower server")
    print(f"         FL rounds     : {args.rounds}")
    print(f"         min clients   : {args.clients}")
    print(f"         port          : {port}")
    print(f"         num_classes   : {args.num_classes}")
    print(f"         Compatible with: mobile_client.py (TensorFlow/Keras)")
    print()
    
    # Create TensorFlow model
    model = create_model(num_classes=args.num_classes)
    init_params = ndarrays_to_parameters(get_parameters(model))
    
    # Strategy
    strategy = MobileFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=init_params,
    )
    
    # Start server
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    
    print("\n[Server] Training complete. Log saved to mobile_round_log.json")


if __name__ == "__main__":
    main()
