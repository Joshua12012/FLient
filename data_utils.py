"""
data_utils.py
-------------
Realistic edge-device data scenario using a synthetic image dataset
that mirrors the MNIST structure (28×28 grayscale, 10 classes).

Key idea – Non-IID Partitioning (Dirichlet Distribution):
  In the real world, edge devices (phones, sensors) collect data from
  their local environment which is NOT uniformly distributed across
  classes.  E.g. a phone in a hospital captures medical digits while a
  retail device captures price tags.

  We simulate this with a Dirichlet(α) distribution.
  α → 0  :  each client gets samples from only 1–2 classes (extreme non-IID)
  α → ∞  :  IID (balanced across all classes)

  Default α = 0.5  gives a moderately heterogeneous split – a good
  representation of real federated edge deployments.

Note on dataset:
  We use a synthetic dataset generated to resemble MNIST.
  Each class has characteristic pixel patterns so classification
  is meaningful even with synthetic data.  In production replace
  SyntheticMNISTLike() with torchvision.datasets.MNIST(download=True).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from typing import List, Tuple, Dict
import os


# ---------------------------------------------------------------------------
# Synthetic Dataset (mimics MNIST structure: 28×28, 10 classes)
# ---------------------------------------------------------------------------

def _make_synthetic(n_samples: int = 60000, n_classes: int = 10,
                    seed: int = 0) -> TensorDataset:
    """
    Generate synthetic 28×28 greyscale images with class-specific structure.
    Each class has a distinctive spatial pattern so a CNN can actually learn.
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // n_classes
    images, labels = [], []

    for c in range(n_classes):
        # Each class: a bright region in a class-specific quadrant + noise
        imgs = rng.random((n_per_class, 28, 28), dtype=np.float32) * 0.3
        # Add a structured bright patch
        row_start = (c % 5) * 5
        col_start = (c // 5) * 14
        imgs[:, row_start:row_start+8, col_start:col_start+12] += 0.7
        imgs = np.clip(imgs, 0.0, 1.0)
        # Normalise same as MNIST
        imgs = (imgs - 0.1307) / 0.3081
        images.append(imgs[:, np.newaxis, :, :])   # add channel dim
        labels.append(np.full(n_per_class, c, dtype=np.int64))

    X = torch.tensor(np.concatenate(images), dtype=torch.float32)
    y = torch.tensor(np.concatenate(labels), dtype=torch.long)
    return TensorDataset(X, y)


class _TargetsWrapper:
    """Thin wrapper so TensorDataset has a .targets attribute (needed by partitioner)."""
    def __init__(self, ds: TensorDataset):
        self._ds = ds
        self.targets = ds.tensors[1]

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


_TRAIN_DS: _TargetsWrapper = None
_TEST_DS:  _TargetsWrapper = None


def load_mnist(train: bool = True) -> _TargetsWrapper:
    """Returns the synthetic dataset (train: 60000 samples, test: 10000)."""
    global _TRAIN_DS, _TEST_DS
    if train:
        if _TRAIN_DS is None:
            _TRAIN_DS = _TargetsWrapper(_make_synthetic(60000, seed=0))
        return _TRAIN_DS
    else:
        if _TEST_DS is None:
            _TEST_DS = _TargetsWrapper(_make_synthetic(10000, seed=99))
        return _TEST_DS


# ---------------------------------------------------------------------------
# Non-IID Dirichlet partitioning
# ---------------------------------------------------------------------------

def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition dataset indices across `num_clients` using Dirichlet(alpha).

    Returns
    -------
    List of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        labels = targets.numpy()
    else:
        labels = np.array(targets)
    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices: Dict[int, np.ndarray] = {
        c: np.where(labels == c)[0] for c in range(num_classes)
    }

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c, idxs in class_indices.items():
        rng.shuffle(idxs)
        # Dirichlet proportions for this class across clients
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        # Convert to absolute counts
        counts = (proportions * len(idxs)).astype(int)
        # Fix rounding: add remainder to last client
        counts[-1] += len(idxs) - counts.sum()

        ptr = 0
        for client_id, cnt in enumerate(counts):
            client_indices[client_id].extend(idxs[ptr: ptr + cnt].tolist())
            ptr += cnt

    return client_indices


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def get_client_loaders(
    num_clients: int = 5,
    alpha: float = 0.5,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Returns
    -------
    train_loaders : list of DataLoaders, one per client
    test_loader   : global test DataLoader
    """
    train_ds = load_mnist(train=True)
    test_ds  = load_mnist(train=False)

    partitions = dirichlet_partition(train_ds, num_clients, alpha, seed)

    train_loaders = []
    for idxs in partitions:
        subset = Subset(train_ds._ds, idxs)
        loader = DataLoader(subset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
        train_loaders.append(loader)

    test_loader = DataLoader(test_ds._ds, batch_size=256,
                             shuffle=False, num_workers=0)

    return train_loaders, test_loader


# ---------------------------------------------------------------------------
# Statistics helpers (for the analysis report)
# ---------------------------------------------------------------------------

def partition_stats(
    dataset,
    partitions: List[List[int]],
) -> List[Dict[int, int]]:
    """
    For each client return a dict {class: count}.
    """
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        labels = targets.numpy()
    else:
        labels = np.array(targets)
    stats = []
    for idxs in partitions:
        client_labels = labels[idxs]
        counts = {int(c): int((client_labels == c).sum())
                  for c in np.unique(client_labels)}
        stats.append(counts)
    return stats


if __name__ == "__main__":
    train_ds = load_mnist()
    partitions = dirichlet_partition(train_ds, num_clients=5, alpha=0.5)
    stats = partition_stats(train_ds, partitions)

    print(f"{'Client':<10}", end="")
    for c in range(10):
        print(f"{'Cls'+str(c):>7}", end="")
    print(f"{'Total':>8}")
    print("-" * 82)

    for i, s in enumerate(stats):
        total = sum(s.values())
        print(f"Client {i:<3}", end="")
        for c in range(10):
            print(f"{s.get(c, 0):>7}", end="")
        print(f"{total:>8}")
