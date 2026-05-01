#!/data/data/com.termux/files/usr/bin/env bash
# termux_setup.sh
# Run this ONCE on each Android phone after installing Termux from F-Droid.
#
# Usage:
#   bash termux_setup.sh
#
# After this completes, run adaptive_serving.py to find your model tier,
# then run client.py with the command it prints.

set -e

echo "============================================"
echo "  Termux FL client setup"
echo "============================================"

# 1. Update package lists
echo "[1/7] Updating packages..."
pkg update -y && pkg upgrade -y

# 2. Install system dependencies
echo "[2/7] Installing system packages..."
pkg install -y python git clang libffi openssl

# 3. Upgrade pip, setuptools, wheel
echo "[3/7] Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# 4. Install required Python packages
echo "[4/7] Installing Python dependencies..."
python -m pip install flwr numpy psutil

# 5. Optional: install PyTorch if a compatible wheel is available
echo "[5/7] Installing PyTorch (optional)..."
python -m pip install --upgrade setuptools wheel
if python -c "import platform, sys; print(platform.machine())" | grep -q -E 'aarch64|arm64'; then
    echo "Detected ARM64/Android architecture. Attempting PyTorch CPU wheel..."
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || true
else
    echo "Skipping PyTorch install: unsupported architecture or wheel not available."
fi

# 6. Create project directory
echo "[6/7] Preparing project directory..."
cd ~
if [ ! -d "fl_project" ]; then
    mkdir -p fl_project
    echo "  [!] Put your project files in ~/fl_project"
    echo "  Either: git clone <your repo>"
    echo "  Or:     scp -r user@<laptop-ip>:/path/to/fl_project ~/fl_project"
fi

# 7. Final message
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Put your project files in ~/fl_project"
echo "   (git clone or scp from your laptop)"
echo ""
echo "2. Find your WiFi IP on laptop:"
echo "   Linux/Mac:  ip addr | grep 192.168"
echo "   Windows:    ipconfig | findstr IPv4"
echo ""
echo "3. Run adaptive_serving.py to pick model tier:"
echo "   cd ~/fl_project"
echo "   python adaptive_serving.py --server <LAPTOP_IP>:8080 --client_id 0 --num_clients 5"
echo ""
echo "4. Copy the command it prints, then run it."
echo ""
