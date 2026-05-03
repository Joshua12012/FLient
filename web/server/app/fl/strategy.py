"""
AdaptiveFedAvg: a Flower FedAvg strategy that records communication-round metrics.

Per round we capture:
- accuracy + avg_loss   (from `aggregate_evaluate`)
- train_time_s          (mean of client-reported `fit` durations)
- comm_time_s           (round wall-clock minus mean train_time)
- comm_bytes            (sum of serialized parameter sizes per client per round)
- samples_per_sec       (total samples / mean train_time)
- mem_mb                (server RSS at end of round, via psutil)
- num_clients

After every aggregate_fit we also persist the global model (Keras + TFLite float16)
so the Gradio "Write" tab can adaptively download the right artifact.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .metrics import MetricsStore, RoundMetric, get_store
from .inference import save_round_artifacts
from .model import get_initial_weights


def _process_mem_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _params_size_bytes(params: Parameters) -> int:
    """Sum the byte length of every tensor in a Parameters message."""
    return sum(len(t) for t in params.tensors)


class AdaptiveFedAvg(FedAvg):
    """
    FedAvg with per-round metrics, communication accounting, and snapshot saving.

    Args:
        tier: which model tier this strategy is aggregating (drives artifact filenames)
        store: shared metrics store (defaults to module singleton)
        target_acc: accuracy at which "time_to_converge" is recorded
    """

    def __init__(
        self,
        *,
        tier: str = "standard",
        store: Optional[MetricsStore] = None,
        target_acc: float = 0.70,
        **kwargs,
    ) -> None:
        # Provide initial parameters so Flower doesn't ask a client for them.
        if "initial_parameters" not in kwargs:
            kwargs["initial_parameters"] = ndarrays_to_parameters(get_initial_weights(tier))
        super().__init__(**kwargs)

        self.tier = tier
        self.store = store or get_store()
        self.store.set_target(target_acc)

        # Per-round bookkeeping populated in configure_fit/aggregate_fit/aggregate_evaluate.
        self._round_start_ts: Dict[int, float] = {}
        self._round_train_times: Dict[int, List[float]] = {}
        self._round_samples: Dict[int, int] = {}
        self._round_comm_bytes: Dict[int, int] = {}
        self._round_num_clients: Dict[int, int] = {}
        self._round_metric_partial: Dict[int, RoundMetric] = {}

    # ------------------------------------------------------------------
    # configure_fit / configure_evaluate: record the start time of each round
    # ------------------------------------------------------------------
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self._round_start_ts[server_round] = time.time()
        self._round_train_times[server_round] = []
        self._round_samples[server_round] = 0
        self._round_comm_bytes[server_round] = 0

        # Communication bytes sent OUT (broadcast): n_clients * size(params).
        broadcast_size = _params_size_bytes(parameters)
        clients_fit_pairs = super().configure_fit(server_round, parameters, client_manager)
        self._round_num_clients[server_round] = len(clients_fit_pairs)
        self._round_comm_bytes[server_round] += broadcast_size * len(clients_fit_pairs)

        # Pass round number + tier through to the client config.
        for _, fit_ins in clients_fit_pairs:
            fit_ins.config["server_round"] = server_round
            fit_ins.config["tier"] = self.tier
        return clients_fit_pairs

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return super().configure_evaluate(server_round, parameters, client_manager)

    # ------------------------------------------------------------------
    # aggregate_fit: weighted average + record train metrics + snapshot model
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Tally client-reported metrics + uplink communication bytes.
        for _, fit_res in results:
            self._round_samples[server_round] += int(fit_res.num_examples)
            self._round_comm_bytes[server_round] += _params_size_bytes(fit_res.parameters)
            train_time = float(fit_res.metrics.get("train_time", 0.0))
            self._round_train_times[server_round].append(train_time)

        aggregated_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            try:
                weights = parameters_to_ndarrays(aggregated_params)
                save_round_artifacts(self.tier, weights)
            except Exception as exc:
                print(f"[strategy] snapshot failed (round {server_round}): {exc}")

        # Compute partial round metric (accuracy/loss filled in by aggregate_evaluate).
        train_times = self._round_train_times[server_round]
        mean_train = float(np.mean(train_times)) if train_times else 0.0
        wall_clock = time.time() - self._round_start_ts[server_round]
        comm_time = max(0.0, wall_clock - mean_train)
        total_samples = self._round_samples[server_round]
        sps = (total_samples / mean_train) if mean_train > 0 else 0.0

        self._round_metric_partial[server_round] = RoundMetric(
            round=server_round,
            accuracy=0.0,
            avg_loss=0.0,
            samples_per_sec=sps,
            mem_mb=_process_mem_mb(),
            train_time_s=mean_train,
            comm_time_s=comm_time,
            comm_bytes=int(self._round_comm_bytes[server_round]),
            num_clients=self._round_num_clients.get(server_round, len(results)),
            tier=self.tier,
        )

        # If no evaluation step is configured, persist immediately so the chart updates.
        if self.fraction_evaluate == 0.0 and self.min_evaluate_clients == 0:
            self.store.record(self._round_metric_partial.pop(server_round))

        return aggregated_params, agg_metrics

    # ------------------------------------------------------------------
    # aggregate_evaluate: fill in accuracy/loss and persist the round
    # ------------------------------------------------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        loss_aggregated, agg_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Mean accuracy weighted by num_examples.
        total_n = sum(r.num_examples for _, r in results) or 1
        acc = sum(r.num_examples * float(r.metrics.get("accuracy", 0.0)) for _, r in results) / total_n

        partial = self._round_metric_partial.pop(server_round, None)
        if partial is None:
            partial = RoundMetric(round=server_round, tier=self.tier, mem_mb=_process_mem_mb())

        partial.accuracy = float(acc)
        partial.avg_loss = float(loss_aggregated) if loss_aggregated is not None else 0.0
        self.store.record(partial)

        merged: Dict[str, Scalar] = dict(agg_metrics or {})
        merged["accuracy"] = float(acc)
        return loss_aggregated, merged
