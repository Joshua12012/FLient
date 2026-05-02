"""
mobile_server.py — Lightweight Flower Server using PyTorch

Uses PyTorch on server side to match mobile_client.py.
Both client and server use the same framework for compatibility.
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict
import logging
import os

from src.utils.connection_utils import find_available_port, is_port_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileModel(nn.Module):
    """PyTorch CNN model matching mobile_client.py."""
    
    def __init__(self, num_classes=10):
        super(MobileModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_parameters(model):
    """Extract parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """Set model parameters from numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


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
    parser = argparse.ArgumentParser(description="Mobile PyTorch Flower Server")
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
    
    print(f"[Mobile Server] Starting PyTorch Flower server")
    print(f"         FL rounds     : {args.rounds}")
    print(f"         min clients   : {args.clients}")
    print(f"         port          : {port}")
    print(f"         num_classes   : {args.num_classes}")
    print(f"         Compatible with: mobile_client.py (PyTorch)")
    print()
    
    # Create PyTorch model
    model = MobileModel(num_classes=args.num_classes)
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
