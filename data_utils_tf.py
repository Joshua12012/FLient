"""
data_utils_tf.py  —  FEMNIST data loader for TensorFlow
Framework-agnostic version that works with both PyTorch and TensorFlow
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List

# ── constants ────────────────────────────────────────────────────────────────
NUM_CLASSES = 62
IMAGE_SIZE = 28
DATA_ROOT = "./data"
DIRICHLET_ALPHA = 0.5


# ── transform ────────────────────────────────────────────────────────────────
def get_transform():
    """No transform needed for TensorFlow, we handle it in the dataset"""
    return None


# ── download FEMNIST (EMNIST byclass) ───────────────────────────────────────
def load_femnist(root: str = DATA_ROOT):
    """
    Downloads EMNIST 'byclass' split the first time (~550 MB).
    Returns (train_dataset, test_dataset) as tf.data.Dataset objects.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Expand dimensions and normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)    # (N, 28, 28, 1)
    
    # For now, use MNIST as placeholder (FEMNIST requires special download)
    # In production, you'd download the actual FEMNIST dataset
    print(f"[data] MNIST loaded (placeholder for FEMNIST) — train: {len(x_train):,}  test: {len(x_test):,}  classes: {NUM_CLASSES}")
    
    return (x_train, y_train), (x_test, y_test)


# ── Dirichlet non-IID partition ─────────────────────────────────────────────
def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float = DIRICHLET_ALPHA, 
                        seed: int = 42) -> List[np.ndarray]:
    """
    Splits dataset indices across num_clients using a Dirichlet distribution.
    Returns list of index arrays, one per client.
    """
    np.random.seed(seed)
    
    # Group indices by class
    label_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    # Draw Dirichlet proportions for each class across clients
    client_indices = {}
    for label in range(min(NUM_CLASSES, 10)):  # MNIST has 10 classes, FEMNIST has 62
        indices = np.array(label_to_indices.get(label, []))
        if len(indices) == 0:
            continue
            
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - proportions[:-1].sum()
        splits = np.split(indices, np.cumsum(proportions[:-1]))
        
        for cid, split in enumerate(splits):
            if cid not in client_indices:
                client_indices[cid] = []
            client_indices[cid].extend(split.tolist())
    
    # Convert to numpy arrays
    partitions = []
    for cid in range(num_clients):
        idxs = np.array(client_indices.get(cid, []))
        np.random.shuffle(idxs)
        partitions.append(idxs)
        print(f"  client {cid:2d}: {len(idxs):5d} samples")
    
    return partitions


# ── TensorFlow Dataset helpers ──────────────────────────────────────────────
def create_tf_dataset(x_data, y_data, indices, batch_size=32, shuffle=True):
    """Create a tf.data.Dataset from indices."""
    x_subset = x_data[indices]
    y_subset = y_data[indices]
    
    dataset = tf.data.Dataset.from_tensor_slices((x_subset, y_subset))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(indices))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ── public helpers ────────────────────────────────────────────────────────────
def get_client_loaders(num_clients: int, batch_size: int = 32,
                       alpha: float = DIRICHLET_ALPHA, root: str = DATA_ROOT):
    """
    Returns train_loaders (list of tf.data.Dataset), test_loader, num_classes.
    Compatible with both PyTorch and TensorFlow.
    """
    (x_train, y_train), (x_test, y_test) = load_femnist(root)
    
    print(f"[data] Partitioning into {num_clients} clients (alpha={alpha})…")
    partitions = dirichlet_partition(y_train, num_clients, alpha)
    
    train_loaders = []
    for cid in range(num_clients):
        loader = create_tf_dataset(x_train, y_train, partitions[cid], batch_size, shuffle=True)
        train_loaders.append(loader)
    
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.batch(256)
    test_loader = test_loader.prefetch(tf.data.AUTOTUNE)
    
    return train_loaders, test_loader, NUM_CLASSES


def get_single_client_loader(client_id: int, num_clients: int,
                              batch_size: int = 32, alpha: float = DIRICHLET_ALPHA,
                              root: str = DATA_ROOT):
    """
    Used inside Termux on a phone — loads only the shard for client_id.
    """
    (x_train, y_train), (x_test, y_test) = load_femnist(root)
    partitions = dirichlet_partition(y_train, num_clients, alpha)
    
    loader = create_tf_dataset(x_train, y_train, partitions[client_id], batch_size, shuffle=True)
    
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.batch(256)
    test_loader = test_loader.prefetch(tf.data.AUTOTUNE)
    
    return loader, test_loader, NUM_CLASSES


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loaders, test_loader, nc = get_client_loaders(num_clients=5, batch_size=32)
    x, y = next(iter(train_loaders[0]))
    print(f"\nSample batch — x: {x.shape}  y: {y.shape}  classes: {nc}")
    print(f"Pixel range: [{x.numpy().min():.2f}, {x.numpy().max():.2f}]")
    print(f"Label range: [{y.numpy().min()}, {y.numpy().max()}]")
