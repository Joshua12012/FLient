"""
tflite_flower_client.py  —  Lightweight TensorFlow-based Flower client

Uses TensorFlow/Keras for training (convertible to TFLite for deployment).
Works with lightweight dataset for fast mobile loading.
"""

import argparse
import time
import numpy as np
import tensorflow as tf
import flwr as fl
from lightweight_data import get_single_client_loader

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_lightweight_model(num_classes=62):
    """
    Create a small TensorFlow model suitable for mobile.
    Can be converted to TFLite for deployment.
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


def model_to_weights(model):
    """Extract model weights as numpy arrays for Flower."""
    return [w.numpy() for w in model.trainable_weights]


def weights_to_model(model, weights):
    """Set model weights from numpy arrays."""
    weight_shapes = [w.shape for w in model.trainable_weights]
    weight_values = []
    idx = 0
    
    for shape in weight_shapes:
        size = np.prod(shape)
        weight_values.append(weights[idx].reshape(shape))
        idx += 1
    
    model.set_weights(weight_values)


class TFLiteFlowerClient(fl.client.NumPyClient):
    """Flower client using TensorFlow/Keras (convertible to TFLite)."""
    
    def __init__(self, client_id, num_clients, epochs=1):
        self.client_id = client_id
        self.num_clients = num_clients
        self.epochs = epochs
        
        # Load lightweight data
        print(f"[Client {client_id}] Loading lightweight dataset...")
        self.train_loader, self.test_loader, num_classes = get_single_client_loader(
            client_id=client_id,
            num_clients=num_clients,
            batch_size=32
        )
        
        # Convert PyTorch data to numpy for TensorFlow
        self.train_data = self._convert_loader_to_arrays(self.train_loader)
        self.test_data = self._convert_loader_to_arrays(self.test_loader)
        
        # Create model
        self.model = create_lightweight_model(num_classes)
        print(f"[Client {client_id}] Model loaded with {self.model.count_params():,} parameters")
    
    def _convert_loader_to_arrays(self, loader):
        """Convert PyTorch DataLoader to numpy arrays for TensorFlow."""
        all_images = []
        all_labels = []
        
        for images, labels in loader:
            # PyTorch: (B, 1, 28, 28) -> TensorFlow: (B, 28, 28, 1)
            images = images.numpy().transpose(0, 2, 3, 1)
            labels = labels.numpy()
            all_images.append(images)
            all_labels.append(labels)
        
        return np.concatenate(all_images), np.concatenate(all_labels)
    
    def get_parameters(self, config):
        return model_to_weights(self.model)
    
    def fit(self, parameters, config):
        # Update model with global weights
        weights_to_model(self.model, parameters)
        
        # Local training
        t0 = time.time()
        history = self.model.fit(
            self.train_data[0], self.train_data[1],
            epochs=self.epochs,
            batch_size=32,
            verbose=0
        )
        train_time = time.time() - t0
        
        train_loss = history.history['loss'][-1]
        
        print(f"[Client {self.client_id}] loss={train_loss:.4f} time={train_time:.1f}s")
        
        metrics = {
            "train_loss": float(train_loss),
            "train_time": float(train_time),
            "client_id": float(self.client_id),
        }
        
        return model_to_weights(self.model), len(self.train_data[0]), metrics
    
    def evaluate(self, parameters, config):
        weights_to_model(self.model, parameters)
        loss, accuracy = self.model.evaluate(
            self.test_data[0], self.test_data[1],
            verbose=0
        )
        return float(loss), len(self.test_data[0]), {"accuracy": float(accuracy)}
    
    def save_tflite(self, path="model.tflite"):
        """Save model as TFLite for mobile deployment."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"[Client {self.client_id}] Saved TFLite model to {path}")


def main():
    parser = argparse.ArgumentParser(description="Lightweight TFLite Flower Client")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080",
                        help="Server IP:port")
    parser.add_argument("--client_id", type=int, default=0,
                        help="Unique client ID")
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Local epochs per round")
    args = parser.parse_args()
    
    print(f"[Client {args.client_id}] Starting lightweight TFLite Flower client...")
    print(f"  Server: {args.server}")
    print(f"  Epochs: {args.epochs}")
    
    client = TFLiteFlowerClient(
        client_id=args.client_id,
        num_clients=args.num_clients,
        epochs=args.epochs
    )
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )


if __name__ == "__main__":
    main()
