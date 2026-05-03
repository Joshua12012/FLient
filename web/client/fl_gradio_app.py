"""
Federated Learning client (Flower + Gradio + TensorFlow).

Flow per Gradio session:
  1. **Connect** -> spin up a Flower NumPyClient in a background thread that
     speaks gRPC to `flwr_server:8080`. Flower drives the rounds; we just
     respond to fit/evaluate.
  2. **Metrics** tab -> `round_metrics.json` via GET /api/metrics/json + summary, polled every 3s.
  3. **Write** tab -> downloads the latest global artifact for the selected tier
     (TFLite for lite / low-resource devices; .weights.h5 otherwise) and runs
     handwriting inference on a sketch.

Comments aim to be plain-English so the file reads top-to-bottom as a tutorial.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import gradio as gr
import numpy as np
import requests
import tensorflow as tf
from PIL import Image

# Make the server-side `app/fl/*` modules importable from the client.
# The client and server share dataset/model code so behavior matches exactly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "server"))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from app.fl import data as fl_data  # noqa: E402
from app.fl.datasets_config import get_dataset_spec  # noqa: E402
from app.fl.model import build_model, idx_to_char  # noqa: E402

# Flower imports are lazy (inside _start_flower_client) so the UI starts even
# when flwr is being installed for the first time.

# ---------------------------------------------------------------------------
# Local model cache (per Gradio session)
# ---------------------------------------------------------------------------

# Where downloaded inference artifacts live on disk.
_DOWNLOAD_DIR = os.path.join(_THIS_DIR, "downloads")
os.makedirs(_DOWNLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Sketch preprocessing (Gradio Sketchpad -> [1, 28, 28, 1] float tensor)
# ---------------------------------------------------------------------------


def preprocess_sketch(image_input: Any) -> np.ndarray:
    """Robustly convert any Gradio Sketchpad payload to a normalized 28x28 tensor."""
    if image_input is None:
        raise ValueError("No image provided. Draw a character first.")

    # Gradio v4 Sketchpad returns a dict with composite/background/layers fields.
    if isinstance(image_input, dict):
        for key in ("composite", "background"):
            if image_input.get(key) is not None:
                image_input = image_input[key]
                break
        else:
            layers = image_input.get("layers") or []
            if layers:
                image_input = layers[-1]
            else:
                raise ValueError("Empty sketch payload.")

    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 3:
            if arr.shape[-1] >= 3:
                arr = arr[..., :3].mean(axis=-1)
            else:
                arr = arr[..., 0]
        img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = image_input

    img = img.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    if fl_data.needs_sketch_spatial_align():
        arr = fl_data.align_sketch_like_training(arr)
    # Dark strokes on light canvas → invert so ink matches “digit brighter than background”.
    if float(np.mean(arr)) > 0.45:
        arr = 1.0 - arr
    spec = get_dataset_spec()
    arr = (arr - spec.mean) / spec.std
    return np.expand_dims(arr, axis=(0, -1))


# ---------------------------------------------------------------------------
# ClientRuntime: per-session state (Flower client thread, model, logs)
# ---------------------------------------------------------------------------


@dataclass
class ClientRuntime:
    server_url: str = ""           # FastAPI HTTP base, e.g. http://host:8000
    fl_address: str = ""           # Flower gRPC, e.g. host:8080
    client_id: str = ""
    tier: str = "standard"
    num_clients: int = 4           # used for partition slot mapping
    alpha: float = 0.5             # Dirichlet skew across clients

    fl_thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    connected: bool = False

    inference_model: Optional[tf.keras.Model] = None  # used by Write tab
    inference_tflite: Optional[bytes] = None
    inference_tier: Optional[str] = None
    inference_kind: Optional[str] = None  # "weights" | "tflite"

    logs: List[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ---------- log helpers ----------
    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self.lock:
            self.logs.append(f"[{ts}] {msg}")
            self.logs = self.logs[-200:]

    def get_logs_text(self) -> str:
        with self.lock:
            return "\n".join(self.logs)

    def clear_logs(self) -> None:
        with self.lock:
            self.logs.clear()

    # ---------- Flower client lifecycle ----------
    def start_flower_client(self) -> None:
        """Launch (or restart) the Flower NumPyClient on a background thread."""
        if self.fl_thread is not None and self.fl_thread.is_alive():
            self.log("Flower client already running for this session.")
            return

        # Build a fresh stop signal so old threads don't hijack a new session.
        self.stop_event = threading.Event()
        self.fl_thread = threading.Thread(
            target=self._run_flower_client, name=f"flwr-client-{self.client_id}", daemon=True
        )
        self.fl_thread.start()

    def _run_flower_client(self) -> None:
        """Body of the background Flower client thread."""
        try:
            import flwr as fl  # local import: not needed for inference-only flows
        except ImportError as exc:
            self.log(f"flwr not installed: {exc}. Run: pip install -r web/client/requirements.txt")
            return

        # Pick this session's data shard once - shared by every fit/eval call.
        try:
            x_tr, y_tr, x_val, y_val = fl_data.load_partition(
                client_id=self.client_id, num_clients=self.num_clients, alpha=self.alpha
            )
            self.log(f"Loaded data shard: train={len(x_tr)} val={len(x_val)} (tier={self.tier})")
        except Exception as exc:
            self.log(f"Could not load data shard: {exc}")
            return

        train_ds = fl_data.make_dataset(x_tr, y_tr, batch_size=64, shuffle=True)
        val_ds = fl_data.evaluate_dataset(x_val, y_val, batch_size=256)
        model = build_model(self.tier)

        runtime = self  # capture self for the inner class

        class _GradioFlowerClient(fl.client.NumPyClient):
            def get_parameters(self, config):
                return model.get_weights()

            def set_parameters(self, parameters):
                model.set_weights(parameters)

            def fit(self, parameters, config):
                self.set_parameters(parameters)
                round_num = int(config.get("server_round", 0))
                runtime.log(f"Round {round_num} fit start")

                t0 = time.time()
                history = model.fit(
                    train_ds,
                    epochs=1,
                    verbose=0,
                )
                train_time = time.time() - t0

                loss = float(history.history["loss"][-1])
                acc = float(history.history.get("accuracy", [0.0])[-1])
                runtime.log(
                    f"Round {round_num} fit done: loss={loss:.4f} acc={acc:.4f} "
                    f"samples={len(x_tr)} train_time={train_time:.2f}s"
                )
                return (
                    model.get_weights(),
                    int(len(x_tr)),
                    {
                        "train_time": float(train_time),
                        "loss": loss,
                        "accuracy": acc,
                        "tier": runtime.tier,
                    },
                )

            def evaluate(self, parameters, config):
                self.set_parameters(parameters)
                t0 = time.time()
                loss, acc = model.evaluate(val_ds, verbose=0)
                eval_time = time.time() - t0
                runtime.log(
                    f"Eval: loss={loss:.4f} acc={acc:.4f} val_n={len(x_val)} t={eval_time:.2f}s"
                )
                return float(loss), int(len(x_val)), {"accuracy": float(acc)}

        # Start the Flower gRPC client - blocks until the server's rounds finish.
        self.connected = True
        try:
            self.log(f"Connecting to Flower server {self.fl_address} ...")
            fl.client.start_client(
                server_address=self.fl_address,
                client=_GradioFlowerClient().to_client(),
                grpc_max_message_length=int(1024 * 1024 * 256),
            )
            self.log("Flower client finished. Ready for inference / new session.")
        except Exception as exc:
            self.log(f"Flower client error: {exc}")
        finally:
            self.connected = False

    # ---------- Adaptive inference download ----------
    def fetch_inference_artifact(self, prefer: str = "auto") -> str:
        """
        Pull the latest global artifact from FastAPI for the selected tier.

        prefer:
        - "auto"    -> TFLite for lite, .weights.h5 otherwise
        - "tflite"  -> always TFLite
        - "weights" -> always .weights.h5
        """
        if not self.server_url:
            raise RuntimeError("Server URL not set. Connect first.")

        if prefer == "auto":
            kind = "tflite" if self.tier == "lite" else "weights"
        else:
            kind = prefer

        url = f"{self.server_url}/api/models/download/{self.tier}/{kind}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            raise RuntimeError(
                f"No {kind} artifact yet for tier '{self.tier}'. Run at least one FL round."
            )
        resp.raise_for_status()

        out_name = f"global_{self.tier}.{('tflite' if kind == 'tflite' else 'weights.h5')}"
        out_path = os.path.join(_DOWNLOAD_DIR, out_name)
        with open(out_path, "wb") as f:
            f.write(resp.content)

        if kind == "tflite":
            self.inference_tflite = resp.content
            self.inference_model = None
        else:
            model = build_model(self.tier)
            model.load_weights(out_path)
            self.inference_model = model
            self.inference_tflite = None

        self.inference_tier = self.tier
        self.inference_kind = kind
        self.log(f"Downloaded {kind} for tier '{self.tier}' ({len(resp.content)} bytes)")
        return out_path

    def _ensure_inference(self) -> None:
        """Lazily download an artifact the first time the user clicks Write."""
        if (
            self.inference_model is None
            and self.inference_tflite is None
        ) or self.inference_tier != self.tier:
            self.fetch_inference_artifact(prefer="auto")

    def predict(self, image_input: Any) -> str:
        """Run inference using either the Keras model or the TFLite interpreter."""
        x = preprocess_sketch(image_input).astype(np.float32)
        self._ensure_inference()

        if self.inference_kind == "tflite" and self.inference_tflite is not None:
            interpreter = tf.lite.Interpreter(model_content=self.inference_tflite)
            interpreter.allocate_tensors()
            input_idx = interpreter.get_input_details()[0]["index"]
            output_idx = interpreter.get_output_details()[0]["index"]
            interpreter.set_tensor(input_idx, x)
            interpreter.invoke()
            probs = interpreter.get_tensor(output_idx)[0]
        elif self.inference_model is not None:
            probs = self.inference_model.predict(x, verbose=0)[0]
        else:
            raise RuntimeError("No inference artifact loaded. Click 'Refresh model'.")

        top3 = np.argsort(probs)[-3:][::-1]
        lines = [
            f"{i+1}. '{idx_to_char(int(idx))}'  ({float(probs[idx]) * 100:.2f}%)"
            for i, idx in enumerate(top3)
        ]
        lines.append(f"(Using {self.inference_kind or 'unknown'} for tier '{self.tier}')")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------


def _runtime(rt: Optional[ClientRuntime]) -> ClientRuntime:
    return rt if rt is not None else ClientRuntime()


def ui_connect(
    server_url: str,
    fl_address: str,
    client_id: str,
    tier: str,
    num_clients: int,
    alpha: float,
    rt: Optional[ClientRuntime],
) -> Tuple[ClientRuntime, str, str]:
    """Wire up server URLs, start Flower client thread."""
    rt = _runtime(rt)
    try:
        rt.server_url = (server_url or "").rstrip("/")
        rt.fl_address = (fl_address or "").strip()
        rt.client_id = (client_id or f"gradio-{uuid.uuid4().hex[:8]}").strip()
        rt.tier = tier
        rt.num_clients = max(1, int(num_clients))
        rt.alpha = float(alpha)

        if not rt.server_url or not rt.fl_address:
            return rt, "Set both Server URL and Flower gRPC address.", rt.get_logs_text()

        # Flower exits after `num_rounds`; FastAPI keeps running but :8080 is dead until /api/fl/ensure.
        try:
            h = requests.get(f"{rt.server_url}/health", timeout=10)
            h.raise_for_status()
            health = h.json()
            srv_ds = (health.get("dataset") or "").strip().lower().replace("-", "_")
            local_ds = os.environ.get("FL_DATASET", "emnist").strip().lower().replace("-", "_")
            if srv_ds and local_ds != srv_ds:
                rt.log(
                    f"Warning: server dataset is {srv_ds!r} but this process has FL_DATASET={local_ds!r}. "
                    f"Set the same env var before starting this client, e.g. set FL_DATASET={srv_ds}"
                )
            if not health.get("flower_alive", False):
                rt.log("Flower gRPC is down (finished past run?). Calling POST /api/fl/ensure ...")
                ens = requests.post(f"{rt.server_url}/api/fl/ensure", timeout=25)
                if ens.ok:
                    rt.log("Flower subprocess (re)started.")
                    time.sleep(1.5)
                else:
                    rt.log(f"/api/fl/ensure failed: HTTP {ens.status_code} {ens.text[:240]}")
            st = requests.get(f"{rt.server_url}/api/fl/status", timeout=10)
            if st.ok:
                meta = st.json()
                srv_tier = meta.get("tier")
                if srv_tier and srv_tier != rt.tier:
                    rt.log(
                        f"Warning: server is training tier '{srv_tier}' but you picked '{rt.tier}'. "
                        f"Use the same tier as `python main.py --tier ...` or weights will not match."
                    )
        except Exception as exc:
            rt.log(f"Server probe failed (continuing): {exc}")

        rt.log(
            f"Connecting client_id='{rt.client_id}' tier='{rt.tier}' "
            f"to gRPC={rt.fl_address}, HTTP={rt.server_url}"
        )
        rt.start_flower_client()
        return rt, f"Connected: {rt.client_id} ({rt.tier})", rt.get_logs_text()
    except Exception as exc:
        rt.log(f"Connect failed: {exc}")
        return rt, f"Connect failed: {exc}", rt.get_logs_text()


def ui_write(image: Any, rt: Optional[ClientRuntime]) -> Tuple[ClientRuntime, str, str]:
    rt = _runtime(rt)
    try:
        pred = rt.predict(image)
        return rt, pred, rt.get_logs_text()
    except Exception as exc:
        rt.log(f"Inference failed: {exc}")
        return rt, f"Inference failed: {exc}", rt.get_logs_text()


def ui_refresh_model(rt: Optional[ClientRuntime]) -> Tuple[ClientRuntime, str, str]:
    rt = _runtime(rt)
    try:
        path = rt.fetch_inference_artifact(prefer="auto")
        return rt, f"Model refreshed: {os.path.basename(path)}", rt.get_logs_text()
    except Exception as exc:
        rt.log(f"Refresh failed: {exc}")
        return rt, f"Refresh failed: {exc}", rt.get_logs_text()


def ui_get_logs(rt: Optional[ClientRuntime]) -> Tuple[ClientRuntime, str]:
    rt = _runtime(rt)
    return rt, rt.get_logs_text()


def ui_clear_logs(rt: Optional[ClientRuntime]) -> Tuple[ClientRuntime, str]:
    rt = _runtime(rt)
    rt.clear_logs()
    return rt, "(logs cleared)"


def ui_new_session(server_url: str, rt: Optional[ClientRuntime]) -> Tuple[ClientRuntime, str, str, str, dict]:
    """
    Reset the *server* metrics + clear local state so the next Connect starts fresh.

    Returns: (new_runtime, status_text, prediction_text, logs_text, metrics_json)
    """
    rt = _runtime(rt)
    base = (server_url or rt.server_url or "").rstrip("/")
    info = ""
    if base:
        try:
            resp = requests.post(f"{base}/api/metrics/reset", timeout=15)
            info = "Server metrics reset OK." if resp.ok else f"Reset HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            info = f"Server reset failed: {exc}"
    else:
        info = "No server URL provided; cleared local state only."

    # Stop the old Flower thread (best effort - it will eventually time out).
    rt.stop_event.set()

    fresh = ClientRuntime()
    fresh.log(info)
    return fresh, "New session ready. Re-Connect to start training.", "", fresh.get_logs_text(), {}


# ---------------------------------------------------------------------------
# Metrics polling (independent of Flower client - just HTTP)
# ---------------------------------------------------------------------------


def ui_poll_metrics(rt: Optional[ClientRuntime]) -> Tuple[ClientRuntime, dict, dict]:
    """Pull full metrics JSON (same as server data/round_metrics.json) + summary."""
    rt = _runtime(rt)
    if not rt.server_url:
        return rt, {}, {"info": "Connect first to populate Server URL."}

    full: dict = {}
    try:
        j = requests.get(f"{rt.server_url}/api/metrics/json", timeout=10)
        if j.ok:
            full = j.json()
    except Exception as exc:
        full = {"error": str(exc)}

    summary: dict = {}
    try:
        sum_resp = requests.get(f"{rt.server_url}/api/metrics/summary", timeout=10)
        if sum_resp.ok:
            summary = sum_resp.json()
    except Exception as exc:
        summary = {"error": str(exc)}

    return rt, full, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FL Gradio Client (Flower)") as demo:
        session = gr.State(value=None)
        gr.Markdown(
            "## Federated Learning Client\n"
            "Connect -> Flower runs the rounds for you. Watch the **Metrics** tab and use **Write** for inference."
        )

        with gr.Row():
            server_url = gr.Textbox(value="http://127.0.0.1:8000", label="FastAPI URL")
            fl_address = gr.Textbox(value="127.0.0.1:8080", label="Flower gRPC address")
            client_id = gr.Textbox(value="", label="Client ID (optional)")
        with gr.Row():
            tier = gr.Dropdown(choices=["lite", "standard", "full"], value="standard", label="Tier")
            num_clients = gr.Number(value=4, label="Total clients (for data partition)", precision=0)
            alpha = gr.Number(value=0.5, label="Dirichlet alpha (lower = more non-IID)")

        with gr.Row():
            connect_btn = gr.Button("Connect", variant="primary")
            new_session_btn = gr.Button("New session (reset metrics)")
            clear_logs_btn = gr.Button("Clear logs")

        status = gr.Textbox(label="Status", interactive=False)

        with gr.Tabs():
            with gr.TabItem("Logs"):
                logs = gr.Textbox(label="Live logs", lines=20, interactive=False)

            with gr.TabItem("Metrics"):
                metrics_file_json = gr.JSON(
                    label="round_metrics.json (GET /api/metrics/json)",
                )
                summary_json = gr.JSON(label="Summary")

            with gr.TabItem("Write (Inference)"):
                gr.Markdown(
                    "Sketch inference matches **FL_DATASET** on the server (emnist letters, fashion classes, kmnist). "
                    "Set the same `FL_DATASET` env on this machine as on the server. "
                    "Model is downloaded from the server (TFLite for lite, Keras weights otherwise)."
                )
                sketch = gr.Sketchpad(label="Draw a sample", type="numpy")
                with gr.Row():
                    write_btn = gr.Button("Write / Predict", variant="primary")
                    refresh_model_btn = gr.Button("Refresh model from server")
                pred_out = gr.Textbox(label="Top predictions", lines=5, interactive=False)

        # Wire callbacks.
        connect_btn.click(
            fn=ui_connect,
            inputs=[server_url, fl_address, client_id, tier, num_clients, alpha, session],
            outputs=[session, status, logs],
        )
        new_session_btn.click(
            fn=ui_new_session,
            inputs=[server_url, session],
            outputs=[session, status, pred_out, logs, metrics_file_json],
        )
        clear_logs_btn.click(
            fn=ui_clear_logs,
            inputs=[session],
            outputs=[session, logs],
        )
        write_btn.click(
            fn=ui_write,
            inputs=[sketch, session],
            outputs=[session, pred_out, logs],
        )
        refresh_model_btn.click(
            fn=ui_refresh_model,
            inputs=[session],
            outputs=[session, status, logs],
        )

        # Background timers for live updates.
        gr.Timer(value=2).tick(
            fn=ui_get_logs, inputs=[session], outputs=[session, logs]
        )
        gr.Timer(value=3).tick(
            fn=ui_poll_metrics,
            inputs=[session],
            outputs=[session, metrics_file_json, summary_json],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
