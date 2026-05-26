# FLient — Federated Learning Client & Server

This repository contains a simple federated learning demo split into two main components:

- `federated_server/` — Python server components (FastAPI + Flower orchestration).
- `fl-client/` — JavaScript client app (mobile/web client, entry `App.js`).

The goal is to provide an easy local workflow to run the server and connect one or more clients to perform federated training.

**Quick Links**

- Server entry: `federated_server/app/main.py`
- Aggregator: `federated_server/app/aggregator.py`
- Client entry: `fl-client/App.js` (also `fl-client/index.js`)
- Server requirements: `federated_server/requirements.txt`

## Requirements

- Python 3.8+ for the server
- Node.js 16+ (or compatible) for the client

## Run locally (recommended)

1. Start the Python server

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r federated_server/requirements.txt
python federated_server/app/main.py
```

2. Start the client

```bash
cd fl-client
npm install
npm start
```

The client will attempt to connect to the server's API/Flower endpoints. If running both on the same machine, ensure the client points to the correct host (localhost or your machine IP).

## Project layout

```
federated_server/
  requirements.txt
  app/
    main.py         # server entry (FastAPI + FL orchestration)
    aggregator.py   # aggregation logic
fl-client/
  package.json
  App.js            # client app entry
  index.js
  assets/
README.md
LICENSE
```

## Notes & troubleshooting

- If ports are blocked, open the server port (example: `sudo ufw allow 8080`).
- For Python dependency issues, create a fresh virtualenv and reinstall.
- The client may be an Expo/React Native or web project depending on `package.json`. Use `npm start` to see the available scripts.

If you want, I can also run a quick smoke test (install deps and start server) — should I proceed?
