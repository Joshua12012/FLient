"""
client.py
---------
Flower FL Client – simulates an edge / mobile device.

Each client:
  1. Receives the global model weights from the server.
  2. Trains locally on its private, non-IID data shard.
  3. Sends back the updated weights (no raw data ever leaves the device).

Additional realism:
  - Simulated bandwidth limit  : large weight tensors are "sent" with a
    configurable artificial delay (bandwidth_mbps param).
  - Local epochs configurable  : devices with more compute can do more
    local steps before syncing.
  - Stragglers                 : a small random extra delay can be injected
    to mimic heterogeneous device speeds.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters, NDArrays, Scalar, ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from model import get_model


# ---------------------------------------------------------------------------
# Helper: convert model weights ↔ Flower Parameters
# ---------------------------------------------------------------------------

def get_parameters(model: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    keys   = list(model.state_dict().keys())
    params = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(params, strict=True)


# ---------------------------------------------------------------------------
# EdgeClient
# ---------------------------------------------------------------------------

class EdgeClient(fl.client.NumPyClient):
    """
    Flower NumpyClient representing a single edge device.

    Parameters
    ----------
    client_id     : int   – unique device identifier
    train_loader  : DataLoader – local training data (non-IID shard)
    test_loader   : DataLoader – global test set (for evaluation)
    model_variant : str   – "large" | "medium" | "small"
    local_epochs  : int   – local training iterations before upload
    lr            : float – local SGD learning rate
    bandwidth_mbps: float – simulated upload bandwidth (MB/s); None = instant
    straggler_prob: float – probability of extra 1-2 s delay
    device        : str   – "cpu" or "cuda"
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_variant: str = "large",
        local_epochs: int = 3,
        lr: float = 0.01,
        bandwidth_mbps: float = None,
        straggler_prob: float = 0.1,
        device: str = "cpu",
    ):
        self.client_id      = client_id
        self.train_loader   = train_loader
        self.test_loader    = test_loader
        self.local_epochs   = local_epochs
        self.lr             = lr
        self.bandwidth_mbps = bandwidth_mbps
        self.straggler_prob = straggler_prob
        self.device_str     = device
        self.torch_device   = torch.device(device)

        self.model = get_model(model_variant).to(self.torch_device)
        self.model_variant  = model_variant

        # Metrics logged locally
        self.round_metrics: List[Dict] = []

    # ------------------------------------------------------------------
    # Flower API
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> NDArrays:
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """Local training step."""
        # 1. Sync with global model
        set_parameters(self.model, parameters)

        # 2. Straggler simulation
        if np.random.rand() < self.straggler_prob:
            delay = np.random.uniform(0.5, 1.5)
            time.sleep(delay)

        # 3. Local training
        t0 = time.perf_counter()
        train_loss = self._train(self.local_epochs)
        train_time = time.perf_counter() - t0

        # 4. Bandwidth simulation  (upload delay)
        updated_params = get_parameters(self.model)
        upload_bytes   = sum(w.nbytes for w in updated_params)
        upload_time    = 0.0
        if self.bandwidth_mbps:
            upload_time = upload_bytes / (self.bandwidth_mbps * 1e6)
            time.sleep(upload_time)

        metrics: Dict[str, Scalar] = {
            "client_id":   self.client_id,
            "train_loss":  float(train_loss),
            "train_time":  round(train_time, 4),
            "upload_kb":   round(upload_bytes / 1024, 1),
            "upload_time": round(upload_time, 4),
            "model_variant": self.model_variant,
        }

        self.round_metrics.append(metrics)
        return updated_params, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate global model on local test set."""
        set_parameters(self.model, parameters)
        loss, accuracy = self._evaluate()
        return float(loss), len(self.test_loader.dataset), {
            "accuracy": float(accuracy),
            "client_id": self.client_id,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train(self, epochs: int) -> float:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        total_loss = 0.0
        total_batches = 0
        for _ in range(epochs):
            for images, labels in self.train_loader:
                images = images.to(self.torch_device)
                labels = labels.to(self.torch_device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss   += loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def _evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.torch_device)
                labels = labels.to(self.torch_device)
                outputs  = self.model(images)
                loss     = criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = outputs.max(1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total
        return avg_loss, accuracy
