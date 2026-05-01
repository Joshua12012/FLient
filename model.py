"""
model.py
--------
Neural network model with Hybrid Parallelism support.

Hybrid Parallelism = Data Parallelism + Model Parallelism
- Data Parallelism  : each client trains on its own local shard of data
- Model Parallelism : the model itself is split into a 'feature extractor'
                      (runs on the edge device / "device_part") and a
                      'classifier head' (runs on the server / "server_part").
                      In simulation we keep both on the same machine but
                      demonstrate the split clearly.

Three model variants let the adaptive-serving layer pick the right size:
  LARGE  → full EdgeCNN (for powerful devices / server)
  MEDIUM → pruned version (mid-range devices)
  SMALL  → tiny MLP (very constrained IoT devices)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Full EdgeCNN – split into two parts for model parallelism
# ---------------------------------------------------------------------------

class EdgeCNN_DevicePart(nn.Module):
    """
    Feature extractor that lives on the EDGE DEVICE.
    Receives raw input (1×28×28 for MNIST) and outputs a compact feature
    vector.  In a real deployment this runs on-device; only the small
    feature vector is transmitted to the server.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28×28→28×28
        self.pool  = nn.MaxPool2d(2, 2)                             # →14×14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # →14×14
        # After second pool: 7×7  → flatten → 32×7×7 = 1568 → fc → 128
        self.fc    = nn.Linear(32 * 7 * 7, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x          # shape: (batch, 128)


class EdgeCNN_ServerPart(nn.Module):
    """
    Classifier head that lives on the SERVER (or a more powerful node).
    Receives the 128-d feature vector and produces class logits.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)   # logits


class EdgeCNN_Full(nn.Module):
    """
    Convenience wrapper that chains both parts.
    Used when running the full model locally (federated simulation).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.device_part = EdgeCNN_DevicePart()
        self.server_part = EdgeCNN_ServerPart(num_classes)

    def forward(self, x):
        features = self.device_part(x)
        return self.server_part(features)

    # ---- helpers used by Flower clients --------------------------------
    def get_weights(self):
        return [p.data.cpu().numpy() for p in self.parameters()]

    def set_weights(self, weights):
        for p, w in zip(self.parameters(), weights):
            p.data = torch.tensor(w, dtype=p.dtype)


# ---------------------------------------------------------------------------
# 2. Medium model – for mid-tier edge devices
# ---------------------------------------------------------------------------

class EdgeCNN_Medium(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(8 * 14 * 14, 64)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_weights(self):
        return [p.data.cpu().numpy() for p in self.parameters()]

    def set_weights(self, weights):
        for p, w in zip(self.parameters(), weights):
            p.data = torch.tensor(w, dtype=p.dtype)


# ---------------------------------------------------------------------------
# 3. Small / Tiny model – for severely constrained IoT devices
# ---------------------------------------------------------------------------

class EdgeCNN_Small(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_weights(self):
        return [p.data.cpu().numpy() for p in self.parameters()]

    def set_weights(self, weights):
        for p, w in zip(self.parameters(), weights):
            p.data = torch.tensor(w, dtype=p.dtype)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "large":  EdgeCNN_Full,
    "medium": EdgeCNN_Medium,
    "small":  EdgeCNN_Small,
}

def get_model(variant: str = "large", num_classes: int = 10) -> nn.Module:
    cls = MODEL_REGISTRY[variant]
    return cls(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, cls in MODEL_REGISTRY.items():
        m = cls()
        print(f"[{name:6s}]  params = {count_parameters(m):>8,}")
        x = torch.zeros(4, 1, 28, 28)
        y = m(x)
        print(f"         output shape = {y.shape}")
