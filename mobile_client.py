"""
mobile_client.py — PyTorch Mobile Federated Learning Client

Uses PyTorch (reliable + smaller than TensorFlow).
~40MB vs ~200MB for TensorFlow, well-tested framework.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
import sys
import os
import logging

# Add parent directories to path for imports
sys.path.append(os.path.dirname(__file__))

from src.utils.connection_utils import (
    connect_with_retry,
    resolve_server_address,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileModel(nn.Module):
    """PyTorch CNN model for mobile federated learning."""
    
    def __init__(self, num_classes=10):
        super(MobileModel, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        
        # FC layers
        self.fc1 = nn.Linear(7 * 7 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten (use reshape instead of view to handle non-contiguous tensors)
        x = x.reshape(x.size(0), -1)
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # FC2
        x = self.fc2(x)
        return x


class MobileFlowerClient(fl.client.NumPyClient):
    """
    Flower client using PyTorch.
    Reliable framework, smaller than TensorFlow, supports training.
    """
    
    def __init__(self, client_id, num_clients, epochs=1, data_path=None):
        self.client_id = client_id
        self.num_clients = num_clients
        self.epochs = epochs
        
        print(f"[Mobile Client {client_id}] Initializing PyTorch model...")
        
        # Load or generate data
        self.x_train, self.y_train, self.x_test, self.y_test, num_classes = self._load_data(data_path)
        
        # Convert numpy to PyTorch tensors
        self.x_train = torch.FloatTensor(self.x_train).permute(0, 3, 1, 2)  # NHWC -> NCHW
        self.y_train = torch.LongTensor(self.y_train)
        self.x_test = torch.FloatTensor(self.x_test).permute(0, 3, 1, 2)
        self.y_test = torch.LongTensor(self.y_test)
        
        # Create PyTorch model
        self.model = MobileModel(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        print(f"[Mobile Client {client_id}] Model ready with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _load_data(self, data_path):
        """Load data or generate synthetic data for testing."""
        num_classes = 10
        
        if data_path and os.path.exists(data_path):
            # Load real data
            print(f"[Client {self.client_id}] Loading data from {data_path}")
            with open(data_path, 'rb') as f:
                data = np.load(f, allow_pickle=True)
            return (data['x_train'], data['y_train'], 
                    data['x_test'], data['y_test'], num_classes)
        else:
            # Generate synthetic MNIST-like data for testing
            print(f"[Client {self.client_id}] Generating synthetic data...")
            samples_per_class = 100
            
            x_train = []
            y_train = []
            for c in range(num_classes):
                x = np.random.randn(samples_per_class, 28, 28, 1).astype(np.float32) * 0.1
                y = np.full(samples_per_class, c, dtype=np.int32)
                x_train.append(x)
                y_train.append(y)
            
            x_train = np.concatenate(x_train)
            y_train = np.concatenate(y_train)
            
            # Shuffle
            indices = np.random.permutation(len(x_train))
            x_train = x_train[indices]
            y_train = y_train[indices]
            
            # Split train/test
            split = int(0.8 * len(x_train))
            x_test = x_train[split:]
            y_test = y_train[split:]
            x_train = x_train[:split]
            y_train = y_train[:split]
            
            print(f"[Client {self.client_id}] Generated {len(x_train)} train, {len(x_test)} test samples")
            
            return x_train, y_train, x_test, y_test, num_classes
    
    def get_parameters(self, config):
        """Return model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train locally with global parameters."""
        # Set global weights
        self.set_parameters(parameters)
        
        print(f"[Mobile Client {self.client_id}] Training for {self.epochs} epochs...")
        t0 = time.time()
        
        # Training loop
        batch_size = 32
        n_samples = len(self.x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            indices = torch.randperm(n_samples)
            
            for i in range(n_batches):
                batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                batch_x = self.x_train[batch_idx]
                batch_y = self.y_train[batch_idx]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        train_loss = avg_loss
        train_time = time.time() - t0
        
        print(f"[Mobile Client {self.client_id}] loss={train_loss:.4f} time={train_time:.1f}s")
        
        metrics = {
            "train_loss": float(train_loss),
            "train_time": float(train_time),
            "client_id": float(self.client_id),
        }
        
        return self.get_parameters(config), len(self.x_train), metrics
    
    def evaluate(self, parameters, config):
        """Evaluate global model."""
        self.set_parameters(parameters)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.x_test)
            loss = self.criterion(outputs, self.y_test).item()
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == self.y_test).sum().item() / len(self.y_test)
        
        print(f"[Mobile Client {self.client_id}] Eval: loss={loss:.4f} acc={accuracy:.4f}")
        
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser(description="Mobile PyTorch Flower Client")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080",
                        help="Server IP:port")
    parser.add_argument("--client_id", type=int, default=0,
                        help="Unique client ID")
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Local epochs per round")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data file (optional)")
    args = parser.parse_args()
    
    print(f"[Mobile Client {args.client_id}] PyTorch Federated Learning Client")
    print(f"  Server: {args.server}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Using PyTorch (smaller than TensorFlow)!")
    
    # Resolve server address with auto-discovery
    resolved_server = resolve_server_address(
        args.server,
        auto_discover=False,  # Disable auto-discovery to avoid hangs
        discover_range=(8080, 8100)
    )
    
    if resolved_server != args.server:
        print(f"[Client {args.client_id}] Auto-discovered server at {resolved_server}")
    
    try:
        client = MobileFlowerClient(
            client_id=args.client_id,
            num_clients=args.num_clients,
            epochs=args.epochs,
            data_path=args.data
        )
        
        # Connect with retry
        def connect_fn(server_address):
            return fl.client.start_numpy_client(
                server_address=server_address,
                client=client,
            )
        
        connect_with_retry(
            connect_fn=connect_fn,
            server_address=resolved_server,
            max_retries=3,  # Reduce retries since GUI already verified connection
            auto_discover=False,  # Disable auto-discovery for GUI
            on_connecting=lambda addr: print(f"[Client {args.client_id}] Connecting to {addr}...")
        )
        
    except Exception as e:
        print(f"[Client {args.client_id}] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
