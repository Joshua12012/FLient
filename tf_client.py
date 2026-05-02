"""
tf_client.py — TensorFlow/Keras Flower client

TensorFlow-based federated learning client for mobile devices.
Lighter than PyTorch and works better on Android.

Usage (phone / Termux):
    python tf_client.py --server 192.168.1.100:8080 --client_id 0 --num_clients 10

Usage (simulated on PC):
    python tf_client.py --server 127.0.0.1:8080 --client_id 2 --num_clients 10
"""

import argparse
import time
import random
import numpy as np
import tensorflow as tf
import flwr as fl
from datetime import datetime
from collections import defaultdict

from tf_model import get_tf_model, NUM_CLASSES


# ── logging helpers ─────────────────────────────────────────────────────────────

def log(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


# ── parameter helpers ─────────────────────────────────────────────────────────

def get_parameters(model):
    """Extract model weights as NumPy arrays for Flower."""
    return [w.numpy() for w in model.trainable_weights]


def set_parameters(model, parameters):
    """Set model weights from NumPy arrays from Flower."""
    model.trainable_weights[0].assign(parameters[0])
    for i, w in enumerate(model.trainable_weights[1:]):
        w.assign(parameters[i+1])


# ── upload size estimator ─────────────────────────────────────────────────────

def estimate_upload_kb(model):
    """Approximate the size of model weights if sent over the wire (float32)."""
    total_bytes = 0
    for w in model.trainable_weights:
        # TensorFlow variables use .shape instead of .size
        size = np.prod(w.shape)
        total_bytes += size * 4  # float32 = 4 bytes
    return total_bytes / 1024


# ── TensorFlow data loading (simplified, no torchvision dependency) ────────────

def load_femnist_synthetic(num_samples=2000, num_classes=62):
    """
    Generate synthetic FEMNIST-like data for testing.
    In production, this would load actual FEMNIST data.
    """
    # Generate synthetic grayscale images
    x = np.random.randn(num_samples, 28, 28, 1).astype(np.float32)
    # Generate random labels
    y = np.random.randint(0, num_classes, num_samples)
    return x, y


def create_dataset(x, y, batch_size=32, shuffle=True):
    """Create a tf.data.Dataset from numpy arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_size)
    return dataset


# ── Flower client ─────────────────────────────────────────────────────────────

class FEMNISTClient(fl.client.NumPyClient):
    def __init__(self, client_id, variant, epochs, batch_size, progress_callback=None):
        self.client_id = client_id
        self.variant = variant
        self.epochs = epochs
        self.batch_size = batch_size
        self.progress_callback = progress_callback

        log(f"[Client {client_id}] Initializing with variant={variant}")

        # Load this client's shard of data (synthetic for now)
        log(f"[Client {client_id}] Loading data shard...")
        t0 = time.time()

        # Generate synthetic data for this client (fixed size per client)
        num_samples = 2000
        x, y = load_femnist_synthetic(num_samples=num_samples)
        
        # Split into train/test
        split = int(0.8 * num_samples)
        self.train_x, self.train_y = x[:split], y[:split]
        self.test_x, self.test_y = x[split:], y[split:]
        
        self.train_dataset = create_dataset(self.train_x, self.train_y, batch_size, shuffle=True)
        self.test_dataset = create_dataset(self.test_x, self.test_y, batch_size, shuffle=False)
        
        load_time = time.time() - t0
        log(f"[Client {client_id}] Dataset loaded in {load_time:.2f}s")
        log(f"[Client {client_id}] Training samples: {len(self.train_x)}")
        log(f"[Client {client_id}] Test samples: {len(self.test_x)}")

        # Load model
        log(f"[Client {client_id}] Loading TensorFlow model...")
        self.model = get_tf_model(variant)
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        log(f"[Client {client_id}] Model loaded successfully")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        log(f"[Client {self.client_id}] === Starting training round ===")
        round_num = config.get("round", 0)
        log(f"[Client {self.client_id}] Round {round_num}")

        # Sync global weights
        log(f"[Client {self.client_id}] Syncing global model weights...")
        set_parameters(self.model, parameters)
        log(f"[Client {self.client_id}] Weights synced successfully")

        # Local training with per-epoch callbacks
        log(f"[Client {self.client_id}] Starting local training for {self.epochs} epochs...")
        t0 = time.time()

        # Custom callback for per-epoch progress
        class EpochProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, callback, epochs):
                self.callback = callback
                self.epochs = epochs

            def on_epoch_end(self, epoch, logs=None):
                if self.callback:
                    loss = logs.get('loss', 0.0)
                    self.callback(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss:.4f}")

        callbacks = []
        if self.progress_callback:
            callbacks.append(EpochProgressCallback(self.progress_callback, self.epochs))

        history = self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            verbose=0,
            callbacks=callbacks
        )

        train_time = time.time() - t0
        train_loss = history.history['loss'][-1]
        log(f"[Client {self.client_id}] Training completed in {train_time:.2f}s")

        # Calculate upload size
        upload_kb = estimate_upload_kb(self.model)

        log(
            f"[Client {self.client_id}] "
            f"loss={train_loss:.4f}  "
            f"time={train_time:.1f}s  "
            f"upload={upload_kb:.1f} KB"
        )

        metrics = {
            "train_loss":  float(train_loss),
            "train_time":  float(train_time),
            "upload_kb":   float(upload_kb),
            "client_id":   float(self.client_id),
        }
        log(f"[Client {self.client_id}] Sending update to server...")
        return get_parameters(self.model), len(self.train_x), metrics

    def evaluate(self, parameters, config):
        log(f"[Client {self.client_id}] Starting evaluation...")
        set_parameters(self.model, parameters)
        loss, accuracy = self.model.evaluate(self.test_dataset, verbose=0)
        log(f"[Client {self.client_id}] Evaluation complete - loss: {loss:.4f}, accuracy: {accuracy:.4f}")
        return float(loss), len(self.test_x), {"accuracy": float(accuracy)}


# ── main ──────────────────────────────────────────────────────────────────────

def main(progress_callback=None):
    parser = argparse.ArgumentParser(description="Flower FEMNIST TensorFlow Client")
    parser.add_argument("--server",       type=str,   default="127.0.0.1:8080",
                        help="Server IP:port")
    parser.add_argument("--client_id",    type=int,   default=0,
                        help="Unique ID for this client")
    parser.add_argument("--variant",      type=str,   default="small",
                        choices=["large","medium","small"],
                        help="Model size")
    parser.add_argument("--epochs",       type=int,   default=1,
                        help="Local epochs per round")
    parser.add_argument("--batch_size",   type=int,   default=32)
    args = parser.parse_args()

    log("=" * 60)
    log(f"Flower FEMNIST TensorFlow Client - ID: {args.client_id}")
    log("=" * 60)

    log(f"Target server: {args.server}")
    log(f"Configuration: variant={args.variant}, epochs={args.epochs}, batch_size={args.batch_size}")

    try:
        log("Creating TensorFlow Flower client...")
        client = FEMNISTClient(
            client_id     = args.client_id,
            variant       = args.variant,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            progress_callback=progress_callback,
        )

        log(f"Connecting to Flower server at {args.server}...")
        log("Waiting for server to accept connection...")

        fl.client.start_numpy_client(
            server_address=args.server,
            client=client,
        )

        log("Federated learning session completed successfully")
        log("=" * 60)

    except ConnectionRefusedError:
        log(f"❌ Connection refused - server not reachable at {args.server}")
        log("Make sure the server is running and the address is correct")
        raise
    except TimeoutError:
        log(f"❌ Connection timeout - server did not respond in time")
        raise
    except Exception as e:
        log(f"❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
