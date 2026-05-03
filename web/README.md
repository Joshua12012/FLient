# Web module — Federated Learning with Flower + Gradio

This directory contains a **Python-only** federated-learning system:

- `web/server/`  - FastAPI HTTP server **plus** a [Flower](https://flower.ai) (`flwr`) gRPC server
- `web/client/`  - [Gradio](https://gradio.app) UI that runs a `flwr.client.NumPyClient` in a background thread

The four project goals are mapped 1:1 to code:

| Goal                                      | Where                                                  |
| ----------------------------------------- | ------------------------------------------------------ |
| Federated learning using Flower (`flwr`)  | `web/server/app/fl/strategy.py`, `web/client/fl_gradio_app.py` |
| Data parallelism                          | `web/server/app/fl/data.py` (tf.data AUTOTUNE) + `MirroredStrategy` flag in `model.py` |
| Communication round analysis              | `web/server/app/fl/metrics.py` + `routers/metrics_router.py` |
| Adaptive inference deployment             | `web/server/app/fl/inference.py` + `routers/model_router.py` |

## Folder layout

```text
web/
  client/
    fl_gradio_app.py         Gradio UI -> flwr.client.NumPyClient
    requirements.txt
  server/
    main.py                  FastAPI on :8000 + flwr.server on :8080 (background thread)
    requirements.txt
    app/
      fl/
        data.py              EMNIST byclass loader + Dirichlet partition + tf.data parallel pipeline
        model.py             3 tiered Keras CNNs (lite / standard / full) + MirroredStrategy flag
        strategy.py          AdaptiveFedAvg: records train_time, comm_time, comm_bytes, accuracy, mem
        metrics.py           RoundMetric + `data/round_metrics.json` (Flower subprocess writes; HTTP reloads file)
        datasets_config.py   FL_DATASET: emnist | fashion_mnist | kmnist
        inference.py         Saves Keras .weights.h5 + float16 TFLite per round
      routers/
        metrics_router.py    /api/metrics/{rounds, summary, json, chart.png placeholder, reset}
        model_router.py      /api/models/{tiers, download/{tier}/{weights, tflite}, recommend}
  GRADIO_MIGRATION_GUIDE.md
  README.md                  this file
```

## Run

```powershell
# 1) one-time install
pip install -r web/server/requirements.txt
pip install -r web/client/requirements.txt

# 2) start FastAPI (:8000) + Flower (:8080) in one process
python web/server/main.py --host 0.0.0.0 --port 8000 --fl-port 8080 --rounds 10 --min-clients 2 --tier standard --dataset fashion_mnist

# 3) start one Gradio client per device — use the **same** FL_DATASET as the server
set FL_DATASET=fashion_mnist
python web/client/fl_gradio_app.py
```

In the Gradio UI:
1. Set **FastAPI URL** (e.g. `http://127.0.0.1:8000`) and **Flower gRPC address** (e.g. `127.0.0.1:8080`).
2. Pick a unique **Client ID** per device (or leave blank for auto-generated).
3. Pick a **tier** matching `--tier` on the server, plus **Total clients** + **alpha** for the data partition.
4. Click **Connect**. Training starts automatically when `--min-clients` are connected.
5. Watch **Logs** + **Metrics** (live `round_metrics.json` via `/api/metrics/json`, also on disk under `web/server/data/`).
6. After at least one round, the **Write** tab can download the latest model and run inference.

## Tailscale notes

Replace `127.0.0.1` with your Tailscale IP (`tailscale ip --4`) when phones/other machines connect:
- FastAPI URL: `http://<server-tailscale-ip>:8000`
- Flower gRPC : `<server-tailscale-ip>:8080`
- Open `http://<server-tailscale-ip>:7860` to use Gradio remotely

## Metrics file

- On disk: `web/server/data/round_metrics.json`
- HTTP: `GET http://<server>:8000/api/metrics/json` (same JSON; FastAPI reloads from disk because Flower runs in a subprocess)

## Dataset (`--dataset` / `FL_DATASET`)

| Value             | Classes | Notes                                      |
| ----------------- | ------- | ------------------------------------------ |
| `emnist` (default)| 62      | Letters + digits (tfds)                    |
| `fashion_mnist` | 10      | Fashion-MNIST (Keras), less “MNIST-toy”    |
| `kmnist`        | 10      | Kuzushiji characters (tfds)                |

Server and every client must use the **same** `FL_DATASET` or model heads and data will not match.

## Reset metrics

Use the Gradio **New session (reset metrics)** button or:

```bash
curl -X POST http://<server>:8000/api/metrics/reset
```
