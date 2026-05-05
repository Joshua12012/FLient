# Federated Learning on Edge Devices

This repository contains two mobile setup paths plus a web-based path:

- **Termux** (Android phones using Termux CLI)
- **Android APK** (Kivy mobile app packaging)
- **Web** (Gradio + FastAPI + Flower)

> Current status: only the **web setup is working reliably**. The Termux and Android APK paths are present in the repo but are currently work in progress.

---

## Working Setup: Web Gradio App

Use the web setup to run the currently supported federated learning flow.

### 1) Install dependencies

```bash
pip install -r web/server/requirements.txt
pip install -r web/client/requirements.txt
```

### 2) Start the server

```bash
python web/server/main.py --host 0.0.0.0 --port 8000 --fl-port 8080 --rounds 10 --min-clients 2 --tier standard --dataset fashion_mnist
```

### 3) Start the Gradio client

```bash
set FL_DATASET=fashion_mnist
python web/client/fl_gradio_app.py
```

### 4) Connect in the UI

- FastAPI URL: `http://127.0.0.1:8000`
- Flower gRPC address: `127.0.0.1:8080`
- Click **Connect** in the Gradio UI

The web Gradio client will connect to the server and begin federated training once the required clients are available.

---

## Setup Paths

### Termux setup

This path is intended for Android devices running Termux, but it is not the currently supported route.

### Android APK setup

The `src/gui/kivy_client.py` and Buildozer packaging path are present, but this mobile APK path is still work in progress.

### Web setup

The web setup under `web/` is the recommended and working path today.

---

## Project layout

```
web/
  server/      # FastAPI + Flower server
  client/      # Gradio UI client
src/          # mobile helper modules and local utilities
requirements.txt
android_requirements.txt
README.md
```

---

## Notes

- Do not rely on the Termux or Android APK flow yet.
- Use the web Gradio app for the current working experience.
- If you want to explore mobile later, the mobile code is present but still under development.

---

## Dataset: FEMNIST

- 62 classes: digits (0–9), A–Z, a–z
- ~800,000 samples, 28×28 grayscale
- Non-IID split via Dirichlet(α=0.5) — each client sees different class distributions
- Downloaded automatically from torchvision on first run (~550 MB)

---

## Adaptive serving tiers

| Tier   | RAM    | Cores | Model  | Params |
| ------ | ------ | ----- | ------ | ------ |
| high   | ≥4 GB  | ≥4    | large  | ~2.3M  |
| medium | 2–4 GB | 2–3   | medium | ~820K  |
| low    | <2 GB  | 1     | small  | ~106K  |

`adaptive_serving.py` detects the device and picks the tier automatically.

---

## Troubleshooting

**"Connection refused" on phone**

- Make sure phone and laptop are on the same WiFi network
- Check laptop firewall allows port 8080: `sudo ufw allow 8080`
- Verify laptop IP with `hostname -I`

**PyTorch install fails on Termux**

- Use CPU-only wheel: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- If that fails: `pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu`

**FEMNIST download slow**

- Download once on PC, then copy `./data/` folder to phones via scp
- `scp -r ./data user@<PHONE_IP>:~/fl_project/data`

**Out of memory on phone**

- Run `adaptive_serving.py` — it will pick `small` model automatically
- Reduce batch size: `--batch_size 16`
- Reduce epochs: `--epochs 1`
