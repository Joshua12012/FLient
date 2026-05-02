# Android Build Guide - Federated Learning Client

This guide will help you build a downloadable Android APK for the Federated Learning client application using Kivy and Flower framework.

## Important Note for Windows Users

**Buildozer's Android support is not available on Windows directly.** Buildozer only supports Android builds on Linux and macOS. On Windows, you have two options:

1. **Use GitHub Actions (Recommended)** - Build the APK in the cloud (easiest option)
2. **Use WSL (Windows Subsystem for Linux)** - Run Ubuntu on Windows and build there

This guide primarily covers the GitHub Actions approach.

## Option 1: Build Using GitHub Actions (Recommended)

### Step 1: Push Your Code to GitHub

1. Create a new repository on GitHub
2. Initialize git in your project:

```bash
git init
git add .
git commit -m "Initial commit"
```

3. Push to GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Step 2: Trigger the Build

The GitHub Actions workflow (`.github/workflows/build-android.yml`) will automatically run when you push to the main branch. You can also manually trigger it:

1. Go to your repository on GitHub
2. Click on "Actions" tab
3. Select "Build Android APK" workflow
4. Click "Run workflow"

### Step 3: Download the APK

1. Wait for the build to complete (takes 10-15 minutes for first build)
2. Go to the Actions tab
3. Click on the completed workflow run
4. Scroll down to "Artifacts" section
5. Download the `android-apk` artifact
6. Extract the ZIP file to get the `.apk` file

## Option 2: Build Using WSL (Alternative)

If you prefer to build locally, use WSL:

### Step 1: Install WSL

```powershell
wsl --install
```

Restart your computer and complete Ubuntu setup.

### Step 2: Install Dependencies in WSL

```bash
sudo apt update
sudo apt install -y build-essential git ffmpeg libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev zlib1g-dev
```

### Step 3: Install Python and Buildozer

```bash
sudo apt install python3 python3-pip
pip3 install buildozer
```

### Step 4: Build the APK

```bash
cd /mnt/d/Federated_Learning_Edge
buildozer android debug
```

## Prerequisites for Development

### On Windows (Your PC)

1. **Install Python 3.8 or later** if not already installed
2. **Install Python dependencies**:

```bash
pip install -r requirements.txt
```

## Installing the APK on Android

### Method 1: USB Transfer

1. After the build completes, you'll have an `.apk` file
2. Connect your Android phone via USB
3. Copy the APK to your phone
4. On your phone, enable "Install from unknown sources" in Settings > Security
5. Tap the APK file to install

### Method 2: ADB Install

If you have ADB installed:

```bash
adb install bin/yourapp-0.1-arm64-v8a-debug.apk
```

## Running the Server on Your PC

Before running the Android app, start the Flower server on your PC:

### Step 1: Find Your PC's IP Address

**Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" (e.g., 192.168.1.100)

**Linux/Mac:**
```bash
hostname -I
```

### Step 2: Start the Server

```bash
python server.py --rounds 20 --clients 5 --variant small --alpha 0.5 --port 8080
```

### Step 3: Configure Firewall

Make sure your firewall allows incoming connections on port 8080:

**Windows:**
```bash
netsh advfirewall firewall add rule name="Flower Server" dir=in action=allow protocol=TCP localport=8080
```

**Linux:**
```bash
sudo ufw allow 8080
```

## Using the Android App

1. **Connect to WiFi**: Ensure your Android phone is on the same WiFi network as your PC
2. **Open the App**: Launch the "Federated Learning Client" app
3. **Enter Server Details**:
   - Server IP:Port: Enter your PC's IP address with port 8080 (e.g., 192.168.1.100:8080)
   - Client ID: Enter a unique number for each device (0, 1, 2, etc.)
   - Total clients: Enter the total number of clients participating (must match server)
4. **Start Training**: Tap "Start Federated Learning"
5. **Monitor Progress**: Watch the status log for training progress

## Troubleshooting

### Buildozer Build Fails

**Issue: "Command 'gcc' failed"**
- Install C++ compiler and build tools
- On Windows: Install Visual Studio Build Tools

**Issue: "Java not found"**
- Install Java JDK 11 or later
- Set JAVA_HOME environment variable

**Issue: "Android SDK not found"**
- Let Buildozer download it automatically (first build)
- Or manually install and set ANDROID_HOME environment variable

### App Won't Connect to Server

**Issue: "Connection refused"**
- Ensure phone and PC are on the same WiFi network
- Check firewall settings on PC
- Verify server is running on the correct port
- Try using PC's IP address instead of localhost

**Issue: App crashes on startup**
- Check the Android logcat: `adb logcat`
- Ensure all required Python files are included in the APK
- Check buildozer.spec for missing dependencies

### Training Issues

**Issue: "Client timeout"**
- Increase timeout in server configuration
- Check network stability
- Reduce model complexity (use 'small' variant)

**Issue: "Out of memory"**
- Use smaller batch size
- Reduce local epochs
- Use 'small' model variant

## Customizing the App

### Change App Name and Icon

Edit `buildozer.spec`:
```ini
title = Your App Name
android.icon = path/to/your/icon.png
```

### Add More Dependencies

Edit `buildozer.spec` requirements line:
```ini
# CRITICAL: Use ONLY NumPy (no TensorFlow, no PyTorch!)
# mobile_client.py uses pure NumPy for training - very lightweight!
requirements = python3,kivy,numpy,psutil,pyjnius,android,flwr,requests
```

### ⚠️ CRITICAL: Why Pure NumPy for Android?

**The Problem:**
- PyTorch (`torch`) doesn't build properly for Android with Buildozer
- TensorFlow is too heavy (100+ MB) for mobile APKs
- TFLite is inference-only (doesn't support training!)

**The Solution:**
The app now uses **`mobile_client.py`** - a pure **NumPy** implementation:
- Custom neural network built with only NumPy
- Supports training on device (SGD optimizer)
- Very lightweight (~1-2 MB added to APK)
- No heavy ML frameworks needed!

**Architecture:**
```bash
Server (your PC):  PyTorch model  →  aggregates weights
Mobile (Android):  NumPy model    →  trains locally
```

Both use the same neural network architecture, just different implementations.

If you see this error on your phone:
```
[ERROR] Import Error: No module named 'mobile_client'
```

**Solution:**
1. Ensure `mobile_client.py` is in your project root
2. `buildozer.spec` should have: `requirements = ...,numpy,flwr` (NO torch/tensorflow)
3. Clean previous builds: `buildozer android clean`
4. Rebuild: `buildozer android debug`

### Change App Permissions

Edit `buildozer.spec`:
```ini
android.permissions = INTERNET,ACCESS_NETWORK_STATE,CAMERA
```

## Next Steps

After successfully building and testing the Android app:

1. **Distribute the APK**: Share the APK with other users
2. **Multiple Devices**: Test with multiple Android phones simultaneously
3. **Performance Tuning**: Adjust model size and training parameters based on device capabilities
4. **Monitor Training**: Use the server's round_log.json to track training progress
5. **iOS Version**: Consider using Kivy's iOS toolchain for iOS deployment (requires Mac)

## Additional Resources

- [Buildozer Documentation](https://buildozer.readthedocs.io/)
- [Kivy Android Packaging](https://kivy.org/doc/stable/guide/packaging-android.html)
- [Flower Framework](https://flower.dev/)
- [TensorFlow Lite Android](https://www.tensorflow.org/lite/android)

## Auto-Connection Features

The client and server now include automatic connection handling:

### Server: Auto Port Selection
- If port 8080 is in use, server automatically finds an available port (8081, 8082, etc.)
- Server displays the actual port it's using

### Client: Auto Server Discovery
- Client scans ports 8080-8100 to find the server
- Retries connection up to 15 times with exponential backoff
- Automatically discovers server even if port changed

### Usage Example

1. Start server (auto-finds port):
```bash
python tflite_server.py --port 8080
# If 8080 in use, will use 8081, 8082, etc.
```

2. Client connects (auto-discovers):
```bash
python tflite_flower_client.py --server 192.168.1.100:8080
# Will scan and find server even on different port
```

3. Or use the launcher:
```bash
python run_fl_with_auto_connect.py --mode both --num_clients 3
```

## Quick Reference

**Build APK:**
```bash
buildozer android debug
```

**Start Server (for Android clients):**
```bash
python mobile_server.py --rounds 20 --clients 5 --port 8080
```

**Start Server (for desktop clients):**
```bash
python src/servers/server.py --rounds 20 --clients 5 --variant small --port 8080
```

**Clean Build (if needed):**
```bash
buildozer android clean
buildozer android debug
```

**View Logs:**
```bash
adb logcat
```
