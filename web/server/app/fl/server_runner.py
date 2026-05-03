"""
Standalone Flower runner process.

Why this exists:
- On Windows, starting Flower from a background *thread* can crash with:
  "signal only works in main thread of the main interpreter".
- FastAPI can still control Flower lifecycle by spawning this module in a
  subprocess during app startup/shutdown.
"""

from __future__ import annotations

import argparse

import flwr as fl

from .strategy import AdaptiveFedAvg


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower gRPC server runner")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--tier", choices=["lite", "standard", "full"], default="standard")
    args = parser.parse_args()

    strategy = AdaptiveFedAvg(
        tier=args.tier,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    print(
        f"[FlowerRunner] Starting gRPC server on {args.host}:{args.port} "
        f"(rounds={args.rounds}, min_clients={args.min_clients}, tier={args.tier})"
    )
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        grpc_max_message_length=int(1024 * 1024 * 256),
    )
    print("[FlowerRunner] Flower server finished.")


if __name__ == "__main__":
    main()
