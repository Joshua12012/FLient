# Migration: WebSocket-FL -> Flower (`flwr`)

The previous custom WebSocket FL implementation has been replaced by [Flower](https://flower.ai).
Same Gradio user flow (Connect / observe Metrics / Write), but the wire protocol is
now `flwr` gRPC and the server records rich communication-round analysis.

## What changed

| Old                                       | New                                                       |
| ----------------------------------------- | --------------------------------------------------------- |
| `app/routers/fl_router.py` (WebSocket FL) | Removed — use `flwr.server` started in `main.py` lifespan |
| `app/routers/client_router.py`            | Removed — device monitor folded into metrics              |
| `app/models/femnist_model.py`             | Replaced by `app/fl/model.py` (deeper, BN, GAP)           |
| `app/utils/femnist_data.py`               | Replaced by `app/fl/data.py` (real EMNIST + tf.data)      |
| Gradio "Train" button + WebSocket polling | Removed — Flower drives rounds automatically              |
| TF.js or synthetic data on the client     | Real EMNIST byclass shard via `tensorflow_datasets`       |

## Install

```powershell
pip install -r web/server/requirements.txt
pip install -r web/client/requirements.txt
```

The first time the server / client touches data it will download the EMNIST byclass
dataset (~600 MB) into `web/server/data/`. If `tensorflow_datasets` cannot reach the
internet the loader falls back to MNIST (10 classes) so the demo still works.

## Run

```powershell
python web/server/main.py --rounds 10 --min-clients 2 --tier standard
python web/client/fl_gradio_app.py            # repeat on each device
```

In the Gradio UI:

1. **Connect** with FastAPI URL + Flower gRPC address. The connect button starts a
   background `flwr.client.NumPyClient` thread; you do **not** click "Train" any more.
2. **Metrics** tab updates every 3s — accuracy, loss, train/comm time, comm bytes, memory.
3. **Write** tab pulls the latest snapshot:
   - `lite` tier or low-resource device -> `.tflite` (float16, fast/small)
   - `standard` / `full` -> `.weights.h5` (best accuracy)

## Tailscale

```powershell
tailscale up
tailscale ip --4
# put that IP into Gradio's Server URL + Flower gRPC fields
# example:  http://100.110.12.34:8000   and   100.110.12.34:8080
```

Allow inbound TCP **8000** (FastAPI), **8080** (Flower gRPC), and optionally
**7860** if a phone connects directly to Gradio.

## Reset between experiments

Click **New session (reset metrics)** in Gradio, or:

```powershell
curl -X POST http://<server>:8000/api/metrics/reset
```

This wipes `web/server/data/round_metrics.json` so the next run starts at round 1.

## Troubleshooting

- **`flwr` import fails** — `pip install flwr` (it's already in both requirements files).
- **`tensorflow_datasets` download is slow** — first run only; the dataset is cached.
- **Gradio shows no metrics** — wait until at least one round finishes (each round needs
  `--min-clients` connected clients).
- **Aggregation shape mismatch** — every client + the server must use the same `--tier`
  (the strategy sets the tier in `fit_ins.config["tier"]`, but you still need to pick the
  matching tier in the Gradio dropdown).
