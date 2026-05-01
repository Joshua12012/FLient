"""
server.py
---------
Federated Learning Server using Flower's strategy API.

Strategy: Custom FedAvg (Federated Averaging – McMahan et al. 2017)
  - Aggregates client weight updates by weighted average
    (weight ∝ number of local samples).
  - Logs per-round metrics for the communication analysis module.
  - Supports optional server-side model evaluation.

Communication round tracking:
  Every round the server records:
    • wall-clock time
    • aggregated loss / accuracy reported by clients
    • number of clients that participated
    • total bytes exchanged (estimated from weight sizes)
"""

import time
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, NDArrays,
    ndarrays_to_parameters, parameters_to_ndarrays,
    EvaluateRes,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from model import get_model


# ---------------------------------------------------------------------------
# Communication-round logger
# ---------------------------------------------------------------------------

class RoundLogger:
    """Accumulates per-round metrics; can be serialised to JSON."""

    def __init__(self):
        self.rounds: List[Dict] = []
        self._start: float = time.perf_counter()

    def log(self, rnd: int, **kwargs):
        entry = {
            "round":    rnd,
            "elapsed_s": round(time.perf_counter() - self._start, 3),
            **kwargs,
        }
        self.rounds.append(entry)
        return entry

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.rounds, f, indent=2)
        print(f"[Logger] Saved round log → {path}")


# ---------------------------------------------------------------------------
# Custom FedAvg Strategy
# ---------------------------------------------------------------------------

class EdgeFedAvg(FedAvg):
    """
    Extends Flower's FedAvg with:
      1. Per-round logging (loss, accuracy, clients, bytes).
      2. Server-side global evaluation after every round.
      3. Printing a clear progress banner to stdout.
    """

    def __init__(self, logger: RoundLogger, eval_fn, model_variant: str = "large", **kwargs):
        super().__init__(**kwargs)
        self.logger        = logger
        self.eval_fn       = eval_fn          # callable(weights) → (loss, accuracy)
        self.model_variant = model_variant

    # ------------------------------------------------------------------
    # Aggregate FIT results
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Standard FedAvg aggregation
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is None:
            return aggregated_params, metrics

        # ---- Collect per-client metrics from this round ---------------
        train_losses  = []
        upload_kbs    = []
        num_examples  = []

        for _, fit_res in results:
            m = fit_res.metrics or {}
            if "train_loss" in m:
                train_losses.append(float(m["train_loss"]))
            if "upload_kb" in m:
                upload_kbs.append(float(m["upload_kb"]))
            num_examples.append(fit_res.num_examples)

        avg_train_loss  = float(np.mean(train_losses))   if train_losses else 0.0
        total_upload_kb = float(np.sum(upload_kbs))       if upload_kbs  else 0.0

        # ---- Server-side global evaluation ----------------------------
        weights = parameters_to_ndarrays(aggregated_params)
        server_loss, server_acc = self.eval_fn(weights)

        # ---- Log round ------------------------------------------------
        entry = self.logger.log(
            server_round,
            num_clients      = len(results),
            avg_train_loss   = round(avg_train_loss, 5),
            server_loss      = round(server_loss, 5),
            server_accuracy  = round(server_acc, 5),
            total_upload_kb  = round(total_upload_kb, 2),
            total_samples    = int(np.sum(num_examples)),
        )

        # ---- Banner ---------------------------------------------------
        banner = (
            f"\n{'='*62}\n"
            f"  Round {server_round:>3}  |  Clients: {len(results)}  |  "
            f"Upload: {total_upload_kb:.1f} KB\n"
            f"  Train Loss: {avg_train_loss:.4f}  |  "
            f"Server Loss: {server_loss:.4f}  |  "
            f"Server Acc: {server_acc*100:.2f}%\n"
            f"{'='*62}"
        )
        print(banner)

        return aggregated_params, {
            "server_loss":     server_loss,
            "server_accuracy": server_acc,
        }

    # ------------------------------------------------------------------
    # Aggregate EVALUATE results
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = sum(r.num_examples for _, r in results)
        weighted_loss  = sum(
            r.num_examples * r.loss for _, r in results
        ) / total_examples
        weighted_acc   = sum(
            r.num_examples * (r.metrics.get("accuracy", 0.0))
            for _, r in results
        ) / total_examples

        return float(weighted_loss), {
            "accuracy":    float(weighted_acc),
        }


# ---------------------------------------------------------------------------
# Server-side evaluation function factory
# ---------------------------------------------------------------------------

def make_eval_fn(test_loader, model_variant: str = "large", device: str = "cpu"):
    """
    Returns a callable that loads `weights` into a fresh model and
    evaluates it on the held-out test set.
    """
    import torch.nn as nn

    torch_device = torch.device(device)

    def evaluate(weights: NDArrays) -> Tuple[float, float]:
        model = get_model(model_variant).to(torch_device)
        keys  = list(model.state_dict().keys())
        state = {k: torch.tensor(w) for k, w in zip(keys, weights)}
        model.load_state_dict(state, strict=True)
        model.eval()

        criterion  = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(torch_device), labels.to(torch_device)
                outputs  = model(images)
                loss     = criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = outputs.max(1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

        return total_loss / len(test_loader), correct / total

    return evaluate
