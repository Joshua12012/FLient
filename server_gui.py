"""
server_gui.py — Flower FedAvg server with GUI control

Provides a GUI to:
1. Monitor client connections
2. Confirm when enough clients are connected
3. Start federated learning explicitly
"""

import argparse
import json
import time
import numpy as np
import tensorflow as tf
import flwr as fl
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.server import Server
from typing import List, Tuple, Optional, Dict, Union
from collections import OrderedDict
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

from tf_model import get_tf_model, NUM_CLASSES


# ── helpers ───────────────────────────────────────────────────────────────────

def get_parameters(model):
    """Extract model weights as NumPy arrays for Flower."""
    return [w.numpy() for w in model.trainable_weights]


def set_parameters(model, parameters):
    """Set model weights from NumPy arrays from Flower."""
    model.trainable_weights[0].assign(parameters[0])
    for i, w in enumerate(model.trainable_weights[1:]):
        w.assign(parameters[i+1])


def evaluate_global(model, test_x, test_y):
    """Evaluate TensorFlow model on test data."""
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    return loss, accuracy


# ── logging strategy with connection tracking ─────────────────────────────────

class LoggingFedAvg(FedAvg):
    """FedAvg + per-round logging to round_log.json + connection tracking."""

    def __init__(self, *args, variant="large", test_x=None, test_y=None,
                 on_client_connect=None, on_round_complete=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant     = variant
        self.test_x      = test_x
        self.test_y      = test_y
        self.round_log   = []
        self.start_time  = time.time()
        self.connected_clients = set()
        self.on_client_connect = on_client_connect
        self.on_round_complete = on_round_complete

    def aggregate_fit(self, server_round, results, failures):
        # Track connected clients
        for client_proxy, _ in results:
            self.connected_clients.add(client_proxy.cid)
            if self.on_client_connect:
                self.on_client_connect(len(self.connected_clients))

        aggregated = super().aggregate_fit(server_round, results, failures)

        # Collect metrics from clients
        if results:
            train_losses = [r.metrics.get("train_loss", 0.0) for _, r in results]
            train_times  = [r.metrics.get("train_time", 0.0) for _, r in results]
            upload_kbs   = [r.metrics.get("upload_kb",  0.0) for _, r in results]

            # Server-side global eval
            if aggregated[0] is not None and self.test_x is not None:
                model = get_tf_model(self.variant)
                params = parameters_to_ndarrays(aggregated[0])
                set_parameters(model, params)
                s_loss, s_acc = evaluate_global(model, self.test_x, self.test_y)
            else:
                s_loss, s_acc = 0.0, 0.0

            entry = {
                "round":            server_round,
                "elapsed_s":        round(time.time() - self.start_time, 2),
                "num_clients":      len(results),
                "avg_train_loss":   float(np.mean(train_losses)),
                "avg_train_time_s": float(np.mean(train_times)),
                "total_upload_kb":  float(np.sum(upload_kbs)),
                "server_loss":      float(s_loss),
                "server_accuracy":  float(s_acc),
            }
            self.round_log.append(entry)

            if self.on_round_complete:
                self.on_round_complete(server_round, len(results), s_acc)

            # Save after every round
            with open("round_log.json", "w") as f:
                json.dump(self.round_log, f, indent=2)

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)


# ── weighted-average metric aggregation helpers ───────────────────────────────

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    agg   = {}
    for n, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v * n / total
    return agg


# ── Server GUI ───────────────────────────────────────────────────────────────

class ServerGUI:
    def __init__(self, root, args):
        self.root = root
        self.root.title("Federated Learning Server")
        self.root.geometry("600x500")

        self.args = args
        self.server_thread = None
        self.server_running = False
        self.connected_clients = 0

        self.create_widgets()
        self.log_message("Server GUI initialized")
        self.log_message(f"Configuration: rounds={args.rounds}, min_clients={args.clients}")

    def create_widgets(self):
        # Configuration frame
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(config_frame, text=f"Port: {self.args.port}").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(config_frame, text=f"Rounds: {self.args.rounds}").grid(row=0, column=1, sticky=tk.W)
        ttk.Label(config_frame, text=f"Min Clients: {self.args.clients}").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(config_frame, text=f"Model: {self.args.variant}").grid(row=1, column=1, sticky=tk.W)

        # Connection status frame
        conn_frame = ttk.LabelFrame(self.root, text="Connection Status", padding=10)
        conn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.clients_label = ttk.Label(conn_frame, text="Connected Clients: 0 / 0", font=('Arial', 12, 'bold'))
        self.clients_label.pack()

        self.progress = ttk.Progressbar(conn_frame, length=400, mode='determinate')
        self.progress.pack(pady=5)

        # Control buttons frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.start_btn = ttk.Button(button_frame, text="Start Server", command=self.start_server)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.fl_btn = ttk.Button(button_frame, text="Start Federated Learning", 
                                command=self.start_fl, state=tk.DISABLED)
        self.fl_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop Server", 
                                  command=self.stop_server, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Log area
        log_frame = ttk.LabelFrame(self.root, text="Server Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_area = scrolledtext.ScrolledText(log_frame, height=10, state=tk.DISABLED)
        self.log_area.pack(fill=tk.BOTH, expand=True)

        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).pack(pady=5)

    def log_message(self, message):
        timestamp = time.strftime('%H:%M:%S')
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state=tk.DISABLED)

    def clear_log(self):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.delete(1.0, tk.END)
        self.log_area.config(state=tk.DISABLED)

    def update_client_count(self, count):
        self.connected_clients = count
        self.clients_label.config(text=f"Connected Clients: {count} / {self.args.clients}")
        self.progress['value'] = (count / self.args.clients) * 100

        if count >= self.args.clients:
            self.fl_btn.config(state=tk.NORMAL)
            self.log_message(f"✓ Minimum clients ({self.args.clients}) reached. Ready to start FL!")
        else:
            self.fl_btn.config(state=tk.DISABLED)

    def on_round_complete(self, round_num, num_clients, accuracy):
        self.log_message(f"Round {round_num} complete: {num_clients} clients, accuracy: {accuracy:.4f}")

    def start_server(self):
        if self.server_running:
            return

        self.log_message("Starting Flower server...")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Start server in background thread
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)
        self.server_thread.start()

    def run_server(self):
        self.server_running = True

        # Generate synthetic test data for global evaluation
        self.log_message("Generating synthetic test data...")
        test_x = np.random.randn(1000, 28, 28, 1).astype(np.float32)
        test_y = np.random.randint(0, NUM_CLASSES, 1000)
        self.log_message(f"Test data ready: {len(test_x)} samples")

        # Initial model parameters
        init_model  = get_tf_model(self.args.variant)
        init_params = ndarrays_to_parameters(get_parameters(init_model))

        strategy = LoggingFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=self.args.clients,
            min_available_clients=self.args.clients,
            initial_parameters=init_params,
            fit_metrics_aggregation_fn=weighted_average,
            variant=self.args.variant,
            test_x=test_x,
            test_y=test_y,
            on_client_connect=self.update_client_count,
            on_round_complete=self.on_round_complete,
        )

        try:
            fl.server.start_server(
                server_address=f"0.0.0.0:{self.args.port}",
                config=fl.server.ServerConfig(num_rounds=self.args.rounds),
                strategy=strategy,
            )
            self.log_message("Server completed successfully")
        except Exception as e:
            self.log_message(f"Server error: {e}")
        finally:
            self.server_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def start_fl(self):
        self.log_message("Federated learning will start when server is running...")
        self.log_message("(Note: FL starts automatically when clients connect)")
        self.fl_btn.config(state=tk.DISABLED)

    def stop_server(self):
        self.log_message("Stopping server...")
        self.server_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Server with GUI")
    parser.add_argument("--rounds",   type=int, default=20,    help="Number of FL rounds")
    parser.add_argument("--clients",  type=int, default=5,     help="Min clients per round")
    parser.add_argument("--variant",  type=str, default="large", choices=["large","medium","small"])
    parser.add_argument("--port",     type=int, default=8080,  help="gRPC port")
    parser.add_argument("--alpha",    type=float, default=0.5, help="Dirichlet alpha for data split")
    args = parser.parse_args()

    root = tk.Tk()
    app = ServerGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
