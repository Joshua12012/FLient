"""
mobile_client.py — Pure NumPy Federated Learning Client

Uses ONLY NumPy for neural network training. 
Smallest possible size (~5MB) - guaranteed to build on GitHub Actions.
"""

import argparse
import time
import numpy as np
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


class NumPyModel:
    """
    Pure NumPy neural network for mobile federated learning.
    Tiny size, no heavy ML frameworks needed.
    """
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.input_shape = (28, 28, 1)
        
        # Initialize weights
        self.params = self._init_params()
        
    def _init_params(self):
        """Initialize network parameters with Xavier initialization."""
        params = {
            # Conv1: 1 -> 8 channels (3x3 kernel)
            'conv1_w': np.random.randn(3, 3, 1, 8) * np.sqrt(2.0 / 9),
            'conv1_b': np.zeros(8),
            
            # Conv2: 8 -> 16 channels (3x3 kernel)
            'conv2_w': np.random.randn(3, 3, 8, 16) * np.sqrt(2.0 / 72),
            'conv2_b': np.zeros(16),
            
            # FC1: 7*7*16 -> 64
            'fc1_w': np.random.randn(7 * 7 * 16, 64) * np.sqrt(2.0 / (7*7*16)),
            'fc1_b': np.zeros(64),
            
            # FC2: 64 -> num_classes
            'fc2_w': np.random.randn(64, self.num_classes) * np.sqrt(2.0 / 64),
            'fc2_b': np.zeros(self.num_classes),
        }
        return params
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _conv2d(self, x, w, b):
        """Simple 2D convolution with SAME padding."""
        batch_size, h, w_in, c_in = x.shape
        kh, kw, _, c_out = w.shape
        
        # Pad input (SAME padding)
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2
        x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        
        output = np.zeros((batch_size, h, w_in, c_out))
        
        for i in range(h):
            for j in range(w_in):
                patch = x_padded[:, i:i+kh, j:j+kw, :]
                for co in range(c_out):
                    # Fix: proper broadcasting - sum over all spatial and input channels
                    output[:, i, j, co] = np.sum(patch * w[:, :, :, co], axis=(1, 2, 3)) + b[co]
        
        return output
    
    def _maxpool2d(self, x, pool_size=2):
        """Max pooling."""
        batch_size, h, w, c = x.shape
        new_h, new_w = h // pool_size, w // pool_size
        output = np.zeros((batch_size, new_h, new_w, c))
        
        for i in range(new_h):
            for j in range(new_w):
                patch = x[:, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size, :]
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
        
        return output
    
    def forward(self, x, training=False):
        """Forward pass."""
        self.cache = {}
        
        # Conv1 -> ReLU -> MaxPool
        conv1 = self._conv2d(x, self.params['conv1_w'], self.params['conv1_b'])
        relu1 = self._relu(conv1)
        pool1 = self._maxpool2d(relu1)
        self.cache['conv1'] = conv1
        self.cache['relu1'] = relu1
        self.cache['pool1'] = pool1
        
        # Conv2 -> ReLU -> MaxPool
        conv2 = self._conv2d(pool1, self.params['conv2_w'], self.params['conv2_b'])
        relu2 = self._relu(conv2)
        pool2 = self._maxpool2d(relu2)
        self.cache['conv2'] = conv2
        self.cache['relu2'] = relu2
        self.cache['pool2'] = pool2
        
        # Flatten
        batch_size = pool2.shape[0]
        flat = pool2.reshape(batch_size, -1)
        self.cache['flat'] = flat
        
        # FC1 -> ReLU
        fc1 = flat @ self.params['fc1_w'] + self.params['fc1_b']
        relu3 = self._relu(fc1)
        self.cache['fc1'] = fc1
        self.cache['relu3'] = relu3
        
        # FC2 -> Softmax
        fc2 = relu3 @ self.params['fc2_w'] + self.params['fc2_b']
        output = self._softmax(fc2)
        
        return output
    
    def compute_loss(self, y_pred, y_true):
        """Cross-entropy loss."""
        n_samples = y_pred.shape[0]
        log_likelihood = -np.log(y_pred[range(n_samples), y_true] + 1e-8)
        return np.sum(log_likelihood) / n_samples
    
    def backward(self, x, y, y_pred, lr=0.01):
        """Backward pass with SGD update."""
        batch_size = y_pred.shape[0]
        
        # Softmax gradient
        dy = y_pred.copy()
        dy[range(batch_size), y] -= 1
        dy /= batch_size
        
        # FC2 backward
        dfc2_w = self.cache['relu3'].T @ dy
        dfc2_b = np.sum(dy, axis=0)
        drelu3 = dy @ self.params['fc2_w'].T
        
        # ReLU backward
        dfc1 = drelu3 * self._relu_derivative(self.cache['fc1'])
        
        # FC1 backward
        dfc1_w = self.cache['flat'].T @ dfc1
        dfc1_b = np.sum(dfc1, axis=0)
        
        # Update weights
        self.params['fc2_w'] -= lr * dfc2_w
        self.params['fc2_b'] -= lr * dfc2_b
        self.params['fc1_w'] -= lr * dfc1_w
        self.params['fc1_b'] -= lr * dfc1_b
    
    def train_step(self, x, y, lr=0.01):
        """Single training step."""
        y_pred = self.forward(x, training=True)
        loss = self.compute_loss(y_pred, y)
        self.backward(x, y, y_pred, lr)
        return loss
    
    def fit(self, x, y, epochs=1, batch_size=32, lr=0.01, verbose=False):
        """Train the model."""
        n_samples = x.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.random.permutation(n_samples)
            
            for i in range(n_batches):
                batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                batch_x = x[batch_idx]
                batch_y = y[batch_idx]
                
                loss = self.train_step(batch_x, batch_y, lr)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_batches
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, x, y):
        """Evaluate model."""
        y_pred = self.forward(x)
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y)
        loss = self.compute_loss(y_pred, y)
        return loss, accuracy
    
    def get_weights_as_list(self):
        """Get weights as list for Flower."""
        return [
            self.params['conv1_w'], self.params['conv1_b'],
            self.params['conv2_w'], self.params['conv2_b'],
            self.params['fc1_w'], self.params['fc1_b'],
            self.params['fc2_w'], self.params['fc2_b'],
        ]
    
    def set_weights_from_list(self, weights):
        """Set weights from list."""
        self.params['conv1_w'] = weights[0]
        self.params['conv1_b'] = weights[1]
        self.params['conv2_w'] = weights[2]
        self.params['conv2_b'] = weights[3]
        self.params['fc1_w'] = weights[4]
        self.params['fc1_b'] = weights[5]
        self.params['fc2_w'] = weights[6]
        self.params['fc2_b'] = weights[7]


class MobileFlowerClient(fl.client.NumPyClient):
    """
    Flower client using pure NumPy.
    Tiny size (~5MB), guaranteed to build on GitHub Actions.
    """
    
    def __init__(self, client_id, num_clients, epochs=1, data_path=None):
        self.client_id = client_id
        self.num_clients = num_clients
        self.epochs = epochs
        
        print(f"[Mobile Client {client_id}] Initializing pure NumPy model...")
        
        # Load or generate data
        self.x_train, self.y_train, self.x_test, self.y_test, num_classes = self._load_data(data_path)
        
        # Create pure NumPy model
        self.model = NumPyModel(num_classes=num_classes)
        
        param_count = sum(p.size for p in self.model.get_weights_as_list())
        print(f"[Mobile Client {client_id}] Model ready with {param_count:,} parameters")
    
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
        return self.model.get_weights_as_list()
    
    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays."""
        self.model.set_weights_from_list(parameters)
    
    def fit(self, parameters, config):
        """Train locally with global parameters."""
        # Set global weights
        self.set_parameters(parameters)
        
        print(f"[Mobile Client {self.client_id}] Training for {self.epochs} epochs...")
        t0 = time.time()
        
        # Train using NumPy model
        avg_loss = self.model.fit(
            self.x_train, self.y_train,
            epochs=self.epochs,
            batch_size=32,
            lr=0.01,
            verbose=True
        )
        
        train_time = time.time() - t0
        
        print(f"[Mobile Client {self.client_id}] loss={avg_loss:.4f} time={train_time:.1f}s")
        
        metrics = {
            "train_loss": float(avg_loss),
            "train_time": float(train_time),
            "client_id": float(self.client_id),
        }
        
        return self.get_parameters(config), len(self.x_train), metrics
    
    def evaluate(self, parameters, config):
        """Evaluate global model."""
        self.set_parameters(parameters)
        
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        
        print(f"[Mobile Client {self.client_id}] Eval: loss={loss:.4f} acc={accuracy:.4f}")
        
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser(description="Mobile NumPy Flower Client")
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
    
    print(f"[Mobile Client {args.client_id}] Pure NumPy Federated Learning Client")
    print(f"  Server: {args.server}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Using pure NumPy (~5MB APK) - guaranteed to build!")
    
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
