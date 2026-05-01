# Federated Learning on Edge Devices

### FEMNIST + Flower + Kivy GUI + Optimized Model Splitting

---

## Project structure

```
fl_project/
├── data_utils.py        ← FEMNIST loader + Dirichlet non-IID partition
├── model.py             ← Large / Medium / Small CNN + split inference
├── server.py            ← Flower FedAvg server with round logging
├── client.py            ← Flower edge client (phone or PC)
├── adaptive_serving.py  ← Device profiler → picks model tier + optimized splitting
├── kivy_client.py       ← Kivy GUI for mobile FL client
├── fl_runner.py         ← Single-PC end-to-end simulation
├── comm_analysis.py     ← Round log plotter
├── requirements.txt
├── termux_setup.sh      ← Run once on each Android phone (alternative)
└── README.md
```

---

## New Features

### Kivy GUI for Mobile Clients

Use the Kivy app for a phone-style interface instead of the Termux CLI.

```bash
python kivy_client.py
```

This app supports:

- device profiling and tier selection
- TFLite model loading and inference
- optional Flower client launch from the GUI
- status logging inside the app

### Running the Kivy app

1. Install Kivy and TFLite runtime dependencies:

```bash
pip install kivy
# For Windows/Linux/Mac desktop:
pip install tensorflow
# For Android/Raspberry Pi mobile:
# pip install tflite-runtime  # (if available for your platform)
```

2. Export a small TFLite model first:

```bash
python tf_model.py --variant small --output mobile_model.tflite --quantize
```

3. Start the GUI:

```bash
python kivy_client.py
```

4. Enter the model path, then press `Load Model`.
5. Press `Run Inference` to check the mobile model.

### Packaging for Android

If you want a real Android app, package `kivy_client.py` with Buildozer:

```bash
git clone https://github.com/kivy/buildozer.git
cd buildozer
pip install -e .
cd ../your-project
buildozer init
# Edit buildozer.spec to include kivy and tflite-runtime as requirements
buildozer android debug
```

The phone can then run the generated APK without Termux.

### Optimized Model Splitting

The `adaptive_serving.py` now benchmarks device performance and chooses the optimal model split point to avoid overloading devices while minimizing latency.

Run device profiling:

```bash
python adaptive_serving.py --server 192.168.1.100:8080 --client_id 0
```

It will output the recommended split configuration based on your device's capabilities.

### TFLite Mobile Runtime (Recommended)

For mobile deployment, TensorFlow Lite is a much better runtime than full PyTorch or JAX on-device.

This project now includes:

- `tf_model.py` — TensorFlow model definitions and TFLite export helper
- `tflite_client.py` — lightweight TFLite client runtime skeleton

Recommended workflow:

1. Train or sync a small model on the server (PyTorch or TensorFlow).
2. Export a mobile-friendly variant to `.tflite` using `tf_model.py`.
3. Deploy the `.tflite` file to the phone and run the client with `tflite_client.py` or a Kivy wrapper.

For example:

```bash
python tf_model.py --variant small --output mobile_model.tflite --quantize
python tflite_client.py --model mobile_model.tflite
```

If you want to keep the server training path in PyTorch, use the TensorFlow/TFLite modules only for mobile inference and feature extraction. That keeps the mobile runtime light and stable.

---

## Phase 1 — Test locally on your PC first

```bash
# Install dependencies
pip install -r requirements.txt

# Run full simulation (5 simulated clients, 10 rounds)
python fl_runner.py --rounds 10 --clients 5 --variant large --alpha 0.5

# After it finishes, view the charts
# fl_results.png is saved in the same folder
```

This downloads FEMNIST (~550 MB) on the first run.

---

## Phase 2 — Real phones via Termux

### Step A — Find your laptop's WiFi IP

```bash
# Linux / Mac
ip addr show | grep "inet 192.168"
# or
hostname -I

# Windows
ipconfig | findstr IPv4
```

Note the IP, e.g. `192.168.1.100`

### Step B — Start the server on your laptop

```bash
python server.py \
  --rounds 20 \
  --clients 5 \
  --variant large \
  --alpha 0.5 \
  --port 8080
```

The server waits until enough clients connect before starting round 1.

### Step C — Set up each Android phone

1. Install **Termux** from [F-Droid](https://f-droid.org) (not Play Store — that version is outdated)

2. Open Termux and run:

```bash
bash termux_setup.sh
```

3. Copy your project files to the phone. Options:

```bash
# Option A — scp from laptop (phone and laptop on same WiFi)
# Run this from your LAPTOP:
scp -r ./fl_project user@<PHONE_IP>:~/fl_project

# Option B — git clone (if you pushed to GitHub)
git clone https://github.com/YOUR_USERNAME/fl_project.git ~/fl_project
```

4. On the phone, run the device profiler:

```bash
cd ~/fl_project
python adaptive_serving.py \
  --server 192.168.1.100:8080 \
  --client_id 0 \          # change this for each phone (0, 1, 2, ...)
  --num_clients 5           # must match --clients on server
```

It prints the exact `python client.py ...` command to run. Copy and run it.

### Step D — Run remaining clients on your laptop (simulated)

If you have 2 phones (client 0 and 1) and need 5 clients total, run clients 2–4 on PC:

```bash
# Terminal 2
python client.py --server 127.0.0.1:8080 --client_id 2 --num_clients 5 --simulate

# Terminal 3
python client.py --server 127.0.0.1:8080 --client_id 3 --num_clients 5 --simulate

# Terminal 4
python client.py --server 127.0.0.1:8080 --client_id 4 --num_clients 5 --simulate
```

### Step E — After training completes

```bash
python comm_analysis.py
# Prints summary + saves fl_results.png
```

---

## Quick reference: arguments

### server.py

| Argument    | Default | Description                           |
| ----------- | ------- | ------------------------------------- |
| `--rounds`  | 20      | Number of FL rounds                   |
| `--clients` | 5       | Min clients required per round        |
| `--variant` | large   | Model size (large / medium / small)   |
| `--alpha`   | 0.5     | Dirichlet skew (lower = more non-IID) |
| `--port`    | 8080    | gRPC port                             |

### client.py

| Argument        | Default        | Description                           |
| --------------- | -------------- | ------------------------------------- |
| `--server`      | 127.0.0.1:8080 | Server IP:port                        |
| `--client_id`   | 0              | Unique client number                  |
| `--num_clients` | 5              | Total clients (must match server)     |
| `--variant`     | large          | Model size                            |
| `--epochs`      | 3              | Local epochs per round                |
| `--alpha`       | 0.5            | Dirichlet alpha (must match server)   |
| `--bandwidth`   | 10.0           | Simulated upload Mbps                 |
| `--straggler`   | off            | Enable random straggler delay         |
| `--simulate`    | off            | Enable bandwidth-limited upload delay |

---

## How the hybrid parallelism works

```
Phone                              Server
─────                              ──────
Input image (1×28×28)
    │
    ▼
EdgeCNN_DevicePart
(conv layers, 3136→128)
    │
    │  128-d feature vector
    │  sent over WiFi
    │  (~512 bytes vs 3136 bytes raw)
    │
    └────────────────────────────►  EdgeCNN_ServerPart
                                    (FC layers → 62 classes)
                                    classification result
```

The phone only runs the cheap early layers. The feature vector is 6× smaller
than the raw flattened image, reducing communication cost.

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
