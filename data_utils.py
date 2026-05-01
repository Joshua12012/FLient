"""
data_utils.py  —  FEMNIST data loader with non-IID Dirichlet partitioning
Replaces the synthetic MNIST-like dataset with real FEMNIST data.

FEMNIST: 62 classes (digits 0-9, uppercase A-Z, lowercase a-z)
         28x28 grayscale images, ~800,000 samples, ~3,500 writers
Source:  torchvision EMNIST split='byclass'  (same underlying data as LEAF FEMNIST)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict

# ── constants ────────────────────────────────────────────────────────────────
NUM_CLASSES   = 62          # FEMNIST: 10 digits + 26 upper + 26 lower
IMAGE_SIZE    = 28
DATA_ROOT     = "./data"    # downloaded here on first run
DIRICHLET_ALPHA = 0.5       # lower = more non-IID (try 0.1 for extreme skew)


# ── transform ────────────────────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST/EMNIST mean & std
    ])


# ── download FEMNIST (EMNIST byclass) ────────────────────────────────────────
def load_femnist(root: str = DATA_ROOT):
    """
    Downloads EMNIST 'byclass' split the first time (≈550 MB).
    Returns (train_dataset, test_dataset) as torchvision Dataset objects.

    EMNIST byclass has 62 classes in the same order FEMNIST uses:
        0-9  → digits
        10-35 → A-Z uppercase
        36-61 → a-z lowercase
    """
    transform = get_transform()
    train_ds = datasets.EMNIST(
        root=root, split="byclass", train=True,
        download=True, transform=transform
    )
    test_ds = datasets.EMNIST(
        root=root, split="byclass", train=False,
        download=True, transform=transform
    )
    print(f"[data] FEMNIST loaded — train: {len(train_ds):,}  test: {len(test_ds):,}  classes: {NUM_CLASSES}")
    return train_ds, test_ds


# ── Dirichlet non-IID partition ───────────────────────────────────────────────
def dirichlet_partition(dataset, num_clients: int, alpha: float = DIRICHLET_ALPHA, seed: int = 42):
    """
    Splits dataset indices across num_clients using a Dirichlet distribution.
    alpha controls heterogeneity:
        alpha → ∞  :  IID (each client sees all classes equally)
        alpha = 0.5:  moderate skew  (default)
        alpha = 0.1:  extreme skew (clients mostly see 1-2 classes)

    Returns list of index arrays, one per client.
    """
    np.random.seed(seed)

    # Group indices by class
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[int(label)].append(idx)

    # Draw Dirichlet proportions for each class across clients
    client_indices = defaultdict(list)
    for label, indices in label_to_indices.items():
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Ensure at least 1 sample per client where possible
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - proportions[:-1].sum()   # fix rounding
        splits = np.split(indices, np.cumsum(proportions[:-1]))
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())

    # Convert to numpy arrays, sort for reproducibility
    partitions = []
    for cid in range(num_clients):
        idxs = np.array(client_indices[cid])
        np.random.shuffle(idxs)
        partitions.append(idxs)
        print(f"  client {cid:2d}: {len(idxs):5d} samples")

    return partitions


# ── thin Dataset wrapper for a client shard ──────────────────────────────────
class ClientDataset(Dataset):
    """Wraps a torchvision dataset + index array into a standard Dataset."""

    def __init__(self, base_dataset, indices):
        self.base    = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


# ── public helpers ────────────────────────────────────────────────────────────
def get_client_loaders(num_clients: int, batch_size: int = 32,
                       alpha: float = DIRICHLET_ALPHA, root: str = DATA_ROOT):
    """
    Main entry point used by client.py and fl_runner.py.
    Downloads FEMNIST once, partitions it, and returns DataLoaders.

    Returns:
        train_loaders : list[DataLoader]  — one per client
        test_loader   : DataLoader        — shared global test set
        num_classes   : int               — always 62
    """
    train_ds, test_ds = load_femnist(root)

    print(f"[data] Partitioning into {num_clients} clients (alpha={alpha})…")
    partitions = dirichlet_partition(train_ds, num_clients, alpha)

    train_loaders = []
    for cid in range(num_clients):
        subset = ClientDataset(train_ds, partitions[cid])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)
        train_loaders.append(loader)

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=0, pin_memory=False)

    return train_loaders, test_loader, NUM_CLASSES


def get_single_client_loader(client_id: int, num_clients: int,
                              batch_size: int = 32, alpha: float = DIRICHLET_ALPHA,
                              root: str = DATA_ROOT):
    """
    Used inside Termux on a phone — loads only the shard for client_id.
    The partition is deterministic (same seed) so each phone gets the same shard
    as long as num_clients and alpha match the server config.
    """
    train_ds, test_ds = load_femnist(root)
    partitions = dirichlet_partition(train_ds, num_clients, alpha)
    subset     = ClientDataset(train_ds, partitions[client_id])
    loader     = DataLoader(subset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=0, pin_memory=False)
    return loader, test_loader, NUM_CLASSES


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loaders, test_loader, nc = get_client_loaders(num_clients=5, batch_size=32)
    x, y = next(iter(train_loaders[0]))
    print(f"\nSample batch — x: {x.shape}  y: {y.shape}  classes: {nc}")
    print(f"Pixel range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Label range: [{y.min()}, {y.max()}]")
