"""
tflite_server.py  —  Lightweight TensorFlow-based Flower server

Compatible with tflite_flower_client.py for fast federated learning.
Uses the same lightweight model architecture.
"""

import argparse
import tensorflow as tf
import flwr as fl
from tflite_flower_client import create_lightweight_model, model_to_weights, weights_to_model

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def fit_config(server_round: int):
    """Return training configuration for each round."""
    return {
        "server_round": server_round,
        "local_epochs": 1,
    }


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    
    def evaluate(server_round, parameters, config):
        # Update model with global weights
        weights_to_model(model, parameters)
        
        # For lightweight testing, use a small validation set
        # In production, you'd load actual validation data
        import numpy as np
        val_images = np.random.randn(100, 28, 28, 1).astype(np.float32)
        val_labels = np.random.randint(0, 62, size=(100,)).astype(np.int32)
        
        loss, accuracy = model.evaluate(val_images, val_labels, verbose=0)
        
        return float(loss), {"accuracy": float(accuracy)}
    
    return evaluate


def main():
    parser = argparse.ArgumentParser(description="Lightweight TFLite Flower Server")
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=3,
                        help="Number of federated rounds")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port")
    args = parser.parse_args()
    
    print(f"[Server] Starting lightweight TFLite Flower server...")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Port: {args.port}")
    
    # Initialize global model
    model = create_lightweight_model(num_classes=62)
    initial_parameters = model_to_weights(model)
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=args.num_clients,
        min_fit_clients=max(1, args.num_clients - 1),
        min_evaluate_clients=max(1, args.num_clients - 1),
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(model),
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
    )
    
    # Start server
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
