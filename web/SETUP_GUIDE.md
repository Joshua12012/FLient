# Federated Learning Web Setup Guide

Complete guide to setting up and running the web-based Federated Learning system.

## Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Server (PC)   │◄───────►│   ngrok Tunnel   │◄───────►│  Client (Phone) │
│   FastAPI + TF  │   HTTP  │   (Public URL)   │   HTTP  │  React + TF.js  │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

## Quick Start

### 1. Start the Server

```bash
cd web/server

# Install dependencies
pip install -r requirements.txt

# Start server with ngrok tunnel
python main.py

# Or without ngrok (local only)
python main.py --no-ngrok
```

The server will:
- Start on `http://localhost:8000`
- Auto-create ngrok tunnel (if not disabled)
- Display the public URL for clients

### 2. Start the Client

```bash
cd web/client

# Install dependencies
npm install

# Start development server
npm run dev
```

The client will:
- Start on `http://localhost:5173`
- Proxy API calls to the server
- Support mobile via the network URL

### 3. Connect Clients

**Option A: Local Network**
- Open `http://<server-ip>:5173` on phones
- Same WiFi network required

**Option B: ngrok (Remote)**
- Copy the ngrok URL from server console
- Share with clients anywhere
- Example: `https://abc123.ngrok.io`

**Option C: Production Deployment**
- Deploy server to cloud
- Use custom domain
- Clients connect via HTTPS

## Detailed Setup

### Server Setup

#### Prerequisites
- Python 3.9+
- pip

#### Installation

```bash
cd web/server

# Create virtual environment (recommended)
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Running the Server

```bash
# With auto ngrok tunnel
python main.py

# Without ngrok (local only)
python main.py --no-ngrok

# Custom port
python main.py --port 8080

# With ngrok auth token (for custom domains)
python main.py --ngrok-token YOUR_TOKEN
```

#### Server API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/docs` | GET | Swagger UI |
| `/api/models/tiers` | GET | List model tiers |
| `/api/fl/register` | POST | Register client |
| `/api/fl/status` | GET | FL status |
| `/api/fl/ws/{client_id}` | WS | WebSocket for FL |
| `/api/clients/report` | POST | Report device status |

### Client Setup

#### Prerequisites
- Node.js 18+
- npm or yarn

#### Installation

```bash
cd web/client

# Install dependencies
npm install

# Or use yarn
yarn install
```

#### Running the Client

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

#### Accessing on Mobile

The Vite dev server shows multiple URLs:
```
  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.1.100:5173/  <-- Use this on phones
```

Use the **Network** URL on phones connected to the same WiFi.

## Tunneling Options

### ngrok (Recommended for Development)

**Pros:**
- Easiest setup
- Good latency (~10-50ms)
- Free tier available

**Setup:**
```bash
# Install ngrok
# https://ngrok.com/download

# The server auto-starts ngrok if pyngrok is installed
pip install pyngrok

# Or manually
ngrok http 8000
```

**Free Tier Limitations:**
- 1 concurrent tunnel
- Random URLs
- 40 connections/minute

**Paid ($8/month):**
- Custom domains
- Multiple tunnels
- No rate limits

### Cloudflare Tunnel (Free + Custom Domain)

**Pros:**
- Completely free
- Persistent subdomain with custom domain
- Good for production

**Setup:**
```bash
# Install cloudflared
# https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation

# Run tunnel
cloudflared tunnel --url http://localhost:8000
```

### Tailscale (Lowest Latency)

**Pros:**
- ~1-5ms latency (WireGuard)
- True mesh VPN
- Free for personal use

**Cons:**
- Requires installation on all devices
- More complex setup

**Setup:**
1. Install Tailscale on all devices
2. Run `tailscale up` on each
3. Use Tailscale IP for connections

## Federated Learning Workflow

1. **Server starts** with global model
2. **Clients connect** via WebSocket
3. **Server initiates round** - sends global weights
4. **Clients train locally** with their data
5. **Clients send updates** back to server
6. **Server aggregates** (FedAvg)
7. **Repeat** for N rounds
8. **Distribute final model** to all clients

## Features

### Adaptive Model Tiers

| Tier | Size | Filters | Hidden | Use Case |
|------|------|---------|--------|----------|
| Lite | ~20KB | 4→8 | 32 | Battery < 20%, thermal |
| Standard | ~40KB | 8→16 | 64 | Normal operation |
| Full | ~80KB | 16→32 | 128 | High-end, charging |

### FEMNIST Dataset

- **62 classes:** 10 digits + 26 lowercase + 26 uppercase
- **Input:** 28×28 grayscale
- **Normalization:** `(x - 0.1307) / 0.3081`

### Data Parallelism

Uses Web Workers for multi-core training:
- Lite tier: 1 worker
- Standard: 2 workers
- Full: 4 workers

## Troubleshooting

### Server Issues

**Port already in use:**
```bash
# Server auto-finds available port
# Or specify different port
python main.py --port 8081
```

**TensorFlow not installing:**
```bash
# Use TensorFlow CPU version
pip install tensorflow-cpu
```

### Client Issues

**Cannot connect to server:**
- Check server is running
- Verify URL (http vs https)
- Check firewall settings

**TensorFlow.js errors:**
- Clear browser cache
- Check WebGL support enabled
- Try different browser

**Mobile display issues:**
- Use modern browser (Chrome/Safari)
- Enable JavaScript
- Check viewport meta tag

### ngrok Issues

**Tunnel not starting:**
```bash
# Check ngrok is installed
ngrok --version

# Install pyngrok
pip install pyngrok
```

**Rate limit exceeded:**
- Wait 1 minute
- Upgrade to paid plan
- Use alternative tunnel

## Production Deployment

### Server Deployment

**Option 1: Cloud VM (AWS/GCP/Azure)**
```bash
# Deploy to cloud instance
# Run with --no-ngrok (use load balancer SSL)
python main.py --no-ngrok --port 8000
```

**Option 2: Docker**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py", "--no-ngrok"]
```

**Option 3: Serverless ( limitations apply )**
- WebSocket support needed
- Consider AWS API Gateway + Lambda

### Client Deployment

**Static Hosting:**
```bash
cd web/client
npm run build
# Deploy dist/ to Netlify/Vercel/S3
```

**Important:** Update server URL in client config for production.

## Security Considerations

1. **Use HTTPS in production** (ngrok provides this)
2. **Add authentication** for production server
3. **Rate limiting** on endpoints
4. **Validate model weights** before aggregation
5. **Log client activity** for audit

## Monitoring

- Server logs: Check console output
- Client logs: Browser developer tools
- WebSocket status: Network tab in DevTools
- Device status: Server dashboard at `/dashboard`

## Support

For issues:
1. Check server/client logs
2. Verify network connectivity
3. Test with local deployment first
4. Check firewall settings
5. Ensure compatible browser versions
