"""
lightweight_data.py  —  Small subset of FEMNIST for fast mobile loading

Uses only 2000 samples total (instead of 800k) for quick testing.
Synthetic generation to avoid downloads.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LightweightFEMNIST(Dataset):
    """Synthetic FEMNIST-like dataset with 2000 samples for fast testing."""
    
    def __init__(self, num_samples=2000, num_classes=62, image_size=28, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Generate synthetic grayscale images (28x28)
        self.images = torch.randn(num_samples, 1, image_size, image_size) * 0.5 + 0.5
        self.images = torch.clamp(self.images, 0, 1)
        
        # Generate random labels
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_lightweight_loaders(num_clients=5, batch_size=32, alpha=0.5, seed=42):
    """
    Create lightweight data loaders for federated learning.
    Total dataset: 2000 samples split across clients.
    """
    np.random.seed(seed)
    
    # Create full dataset
    full_dataset = LightweightFEMNIST(num_samples=2000, num_classes=62)
    
    # Simple partition: roughly equal split with some randomness
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    
    partitions = np.array_split(indices, num_clients)
    
    train_loaders = []
    for client_id in range(num_clients):
        client_indices = partitions[client_id]
        subset = torch.utils.data.Subset(full_dataset, client_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        train_loaders.append(loader)
        print(f"Client {client_id}: {len(client_indices)} samples")
    
    # Use full dataset as test set for simplicity
    test_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    
    return train_loaders, test_loader, 62


def get_single_client_loader(client_id=0, num_clients=5, batch_size=32, alpha=0.5, seed=42):
    """
    Get a single client's data loader for mobile deployment.
    Deterministic partition based on seed.
    """
    np.random.seed(seed)
    
    full_dataset = LightweightFEMNIST(num_samples=2000, num_classes=62)
    
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    
    partitions = np.array_split(indices, num_clients)
    client_indices = partitions[client_id]
    
    subset = torch.utils.data.Subset(full_dataset, client_indices)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    
    print(f"Client {client_id}: {len(client_indices)} training samples")
    
    return train_loader, test_loader, 62


if __name__ == "__main__":
    # Quick test
    train_loaders, test_loader, num_classes = get_lightweight_loaders(num_clients=5)
    x, y = next(iter(train_loaders[0]))
    print(f"Batch shape: {x.shape}, Labels: {y.shape}")
    print(f"Num classes: {num_classes}")
