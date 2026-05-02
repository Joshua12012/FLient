"""
mobile_client.py — TensorFlow/Keras Federated Learning Client

Uses TensorFlow/Keras for reliable training on Android.
Works with mobile_server.py for federated learning.
"""

import argparse
import time
import numpy as np
import tensorflow as tf
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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(num_classes=10):
    """
    Create a TensorFlow/Keras CNN model.
    Lightweight but reliable - uses tested framework.
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


class MobileFlowerClient(fl.client.NumPyClient):
    """
    Flower client using TensorFlow/Keras.
    Reliable framework, well-tested, supports training.
    """
    
    def __init__(self, client_id, num_clients, epochs=1, data_path=None):
        self.client_id = client_id
        self.num_clients = num_clients
        self.epochs = epochs
        
        print(f"[Mobile Client {client_id}] Initializing TensorFlow/Keras model...")
        
        # Load or generate data
        self.x_train, self.y_train, self.x_test, self.y_test, num_classes = self._load_data(data_path)
        
        # Create TensorFlow model
        self.model = create_model(num_classes=num_classes)
        print(f"[Mobile Client {client_id}] Model ready with {self.model.count_params():,} parameters")
    
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
        return [w.numpy() for w in self.model.trainable_weights]
    
    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays."""
        weight_shapes = [w.shape for w in self.model.trainable_weights]
        weight_values = []
        idx = 0
        
        for shape in weight_shapes:
            size = np.prod(shape)
            weight_values.append(parameters[idx].reshape(shape))
            idx += 1
        
        self.model.set_weights(weight_values)
    
    def fit(self, parameters, config):
        """Train locally with global parameters."""
        # Set global weights
        self.set_parameters(parameters)
        
        print(f"[Mobile Client {self.client_id}] Training for {self.epochs} epochs...")
        t0 = time.time()
        
        # Train
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=self.epochs,
            batch_size=32,
            verbose=0  # Minimal logging
        )
        
        train_loss = history.history['loss'][-1]
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
        
        loss, accuracy = self.model.evaluate(
            self.x_test, self.y_test,
            verbose=0
        )
        
        print(f"[Mobile Client {self.client_id}] Eval: loss={loss:.4f} acc={accuracy:.4f}")
        
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser(description="Mobile TensorFlow/Keras Flower Client")
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
    
    print(f"[Mobile Client {args.client_id}] TensorFlow/Keras Federated Learning Client")
    print(f"  Server: {args.server}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Using reliable TensorFlow framework!")
    
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
