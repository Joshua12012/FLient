"""
model.py  —  Three model tiers for adaptive serving + split inference
Updated for FEMNIST: NUM_CLASSES = 62

Tiers:
    large   → full EdgeCNN, also split into DevicePart / ServerPart
    medium  → lighter CNN, runs on mid-range phones
    small   → tiny CNN for low-RAM devices

Split inference (hybrid parallelism):
    EdgeCNN_DevicePart  : early layers that run ON the phone
    EdgeCNN_ServerPart  : later layers that run on the server
    The phone sends a 128-d feature vector, not raw pixels, to the server.
    This is your "model parallelism" + communication efficiency story.
"""

import torch
import torch.nn as nn

NUM_CLASSES   = 62   # FEMNIST: digits + upper + lower
FEATURE_DIM   = 128  # size of the intermediate feature vector sent over the wire


# ══════════════════════════════════════════════════════════════════════════════
# LARGE MODEL — split into device + server parts
# ══════════════════════════════════════════════════════════════════════════════

class EdgeCNN_DevicePart(nn.Module):
    """
    Runs on the edge device (phone).
    Input : (B, 1, 28, 28) grayscale image
    Output: (B, 128) feature vector  ← this is what gets sent to the server
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # → (B,32,28,28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # → (B,32,28,28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (B,32,14,14)
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (B,64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → (B,64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (B,64,7,7)
            nn.Dropout2d(0.1),
        )
        self.flatten  = nn.Flatten()                       # → (B, 64*7*7=3136)
        self.compress = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, FEATURE_DIM),                  # → (B, 128)
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.compress(x)
        return x                                           # (B, 128)


class EdgeCNN_ServerPart(nn.Module):
    """
    Runs on the server after receiving the 128-d feature vector.
    Input : (B, 128) feature vector from device
    Output: (B, 62) class logits
    """
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),                  # → (B, 62)
        )

    def forward(self, features):
        return self.classifier(features)


class EdgeCNN_Large(nn.Module):
    """
    Combines both parts for standard federated training
    (no split — full model on one machine).
    """
    def __init__(self):
        super().__init__()
        self.device_part = EdgeCNN_DevicePart()
        self.server_part = EdgeCNN_ServerPart()

    def forward(self, x):
        features = self.device_part(x)
        return self.server_part(features)


# ══════════════════════════════════════════════════════════════════════════════
# MEDIUM MODEL — mid-range phones (2–4 GB RAM)
# ══════════════════════════════════════════════════════════════════════════════

class EdgeCNN_Medium(nn.Module):
    """
    Lighter model: fewer filters, no BatchNorm, single FC layer.
    ~3× fewer parameters than large.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # → (B,16,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (B,16,14,14)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → (B,32,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (B,32,7,7)

            nn.Flatten(),                                  # → (B, 32*7*7=1568)
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),                  # → (B, 62)
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# SMALL MODEL — low-end phones (<2 GB RAM)
# ══════════════════════════════════════════════════════════════════════════════

class EdgeCNN_Small(nn.Module):
    """
    Minimal model: single conv block + one FC.
    Fast to train even on weak CPUs.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),    # → (B,8,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),                               # → (B,8,7,7)

            nn.Flatten(),                                  # → (B, 8*7*7=392)
            nn.Linear(392, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),                  # → (B, 62)
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Factory helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_model(variant: str = "large") -> nn.Module:
    """
    Returns the full (non-split) model for a given variant.
    Used by server.py and simulated clients.
    variant: 'large' | 'medium' | 'small'
    """
    models = {
        "large":  EdgeCNN_Large,
        "medium": EdgeCNN_Medium,
        "small":  EdgeCNN_Small,
    }
    if variant not in models:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(models)}")
    return models[variant]()


def get_split_model():
    """Returns (device_part, server_part) for split inference demo."""
    return EdgeCNN_DevicePart(), EdgeCNN_ServerPart()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28)
    for v in ["large", "medium", "small"]:
        m = get_model(v)
        out = m(x)
        print(f"{v:8s}  params: {count_parameters(m):>8,}  output: {out.shape}")

    print("\n--- split inference demo ---")
    dev, srv = get_split_model()
    feat = dev(x)
    print(f"Device part output (feature vector): {feat.shape}")
    logits = srv(feat)
    print(f"Server part output (logits):         {logits.shape}")
