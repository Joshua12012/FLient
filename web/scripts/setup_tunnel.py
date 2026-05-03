"""
Tunnel Setup Script - Multiple tunneling options

Supports:
1. ngrok (easiest, recommended for development)
2. Cloudflare Tunnel (free, requires domain)
3. Tailscale (lowest latency, mesh VPN)

Usage:
    python setup_tunnel.py --type ngrok --port 8000
    python setup_tunnel.py --type cloudflare --port 8000
    python setup_tunnel.py --type tailscale --port 8000
"""

import argparse
import subprocess
import sys
import os
import json
import time
from typing import Optional, Tuple


def check_command(cmd: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def setup_ngrok(port: int, auth_token: Optional[str] = None) -> Tuple[bool, str]:
    """
    Setup ngrok tunnel.
    
    Pros:
    - Easiest setup (single command)
    - Good documentation
    - HTTP/HTTPS support
    
    Cons:
    - Free tier: random URLs, 1 concurrent tunnel
    - Pro: $8/month for custom domains
    
    Latency: ~10-50ms additional
    """
    print("\n[ngrok] Setting up tunnel...")
    
    try:
        from pyngrok import ngrok
        
        # Set auth token if provided
        if auth_token:
            ngrok.set_auth_token(auth_token)
        
        # Kill existing tunnels
        ngrok.kill()
        
        # Open tunnel
        public_url = ngrok.connect(port, "http")
        
        print(f"✅ ngrok tunnel active!")
        print(f"   Public URL: {public_url}")
        print(f"   Local:      http://localhost:{port}")
        print(f"\n   Share this URL with clients: {public_url}")
        
        return True, str(public_url)
    
    except ImportError:
        print("❌ pyngrok not installed.")
        print("   Install: pip install pyngrok")
        print("   Or download ngrok: https://ngrok.com/download")
        return False, ""
    
    except Exception as e:
        print(f"❌ ngrok error: {e}")
        return False, ""


def setup_cloudflare(port: int) -> Tuple[bool, str]:
    """
    Setup Cloudflare Tunnel (cloudflared).
    
    Pros:
    - Completely free
    - No bandwidth limits
    - Persistent subdomain (with custom domain)
    - Good for production
    
    Cons:
    - Requires Cloudflare account
    - Requires domain for custom subdomain
    - More complex initial setup
    
    Latency: ~10-30ms additional
    """
    print("\n[Cloudflare] Setting up tunnel...")
    
    if not check_command("cloudflared"):
        print("❌ cloudflared not installed.")
        print("   Install:")
        print("     Windows: choco install cloudflared")
        print("     macOS:   brew install cloudflared")
        print("     Linux:   https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation")
        return False, ""
    
    try:
        # Run cloudflared tunnel
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait and extract URL
        time.sleep(3)
        
        # Try to get the URL from output
        # This is a simplified version - in practice, parse the output
        print("✅ Cloudflare tunnel starting...")
        print("   Check the cloudflared output for the public URL")
        print(f"   Local: http://localhost:{port}")
        
        return True, ""
    
    except Exception as e:
        print(f"❌ Cloudflare error: {e}")
        return False, ""


def setup_tailscale() -> Tuple[bool, str]:
    """
    Setup Tailscale mesh VPN.
    
    Pros:
    - Lowest latency (~1-5ms, WireGuard-based)
    - True mesh VPN (direct connections)
    - Free for personal use (20 devices)
    - Static IPs
    - Most secure
    
    Cons:
    - Requires installation on all devices
    - Requires account
    - More complex setup
    - Not a public URL (requires all clients on same tailnet)
    
    Best for: Long-term deployments where all clients can install Tailscale
    """
    print("\n[Tailscale] Checking setup...")
    
    if not check_command("tailscale"):
        print("❌ tailscale not installed.")
        print("   Download: https://tailscale.com/download")
        print("   Then run: tailscale up")
        return False, ""
    
    try:
        # Get Tailscale IP
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            check=True
        )
        tailscale_ip = result.stdout.strip()
        
        print("✅ Tailscale is active!")
        print(f"   Tailscale IP: {tailscale_ip}")
        print(f"   Share this IP with clients on the same tailnet")
        
        return True, tailscale_ip
    
    except subprocess.CalledProcessError:
        print("❌ Tailscale not running.")
        print("   Start with: tailscale up")
        return False, ""
    
    except Exception as e:
        print(f"❌ Tailscale error: {e}")
        return False, ""


def setup_localtunnel(port: int) -> Tuple[bool, str]:
    """
    Setup LocalTunnel (free alternative to ngrok).
    
    Pros:
    - Completely free
    - Simple setup
    
    Cons:
    - Less reliable than ngrok
    - Sometimes unstable
    - No custom domains
    
    Latency: ~50-100ms additional (variable)
    """
    print("\n[LocalTunnel] Setting up...")
    
    # Check if npx is available
    if not check_command("npx"):
        print("❌ npx not found. Install Node.js")
        return False, ""
    
    try:
        print("✅ Starting LocalTunnel...")
        print(f"   Run manually: npx localtunnel --port {port}")
        print("   (Press Ctrl+C and run the above command in a new terminal)")
        
        # Option to start automatically
        proc = subprocess.Popen(
            ["npx", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(5)
        print("   LocalTunnel should be starting...")
        
        return True, ""
    
    except Exception as e:
        print(f"❌ LocalTunnel error: {e}")
        return False, ""


def print_comparison():
    """Print comparison of tunneling options."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                 TUNNELING OPTIONS COMPARISON                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. NGROK (Recommended)                                           ║
║     • Setup:     ⭐ Easiest (pip install pyngrok)                  ║
║     • Latency:   ~10-50ms                                          ║
║     • Cost:      Free tier available, $8/mo for pro               ║
║     • Best for:  Development, quick demos                           ║
║                                                                    ║
║  2. CLOUDFLARE TUNNEL                                               ║
║     • Setup:     Moderate (requires account)                       ║
║     • Latency:   ~10-30ms                                          ║
║     • Cost:      Free (requires domain for custom URL)            ║
║     • Best for:  Production with custom domain                    ║
║                                                                    ║
║  3. TAILSCALE                                                       ║
║     • Setup:     Complex (install on all devices)                  ║
║     • Latency:   ~1-5ms (WireGuard)                                ║
║     • Cost:      Free (personal, 20 devices)                      ║
║     • Best for:  Long-term deployments, lowest latency            ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Setup tunnel for FL Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_tunnel.py --type ngrok --port 8000
  python setup_tunnel.py --type ngrok --port 8000 --token YOUR_TOKEN
  python setup_tunnel.py --type cloudflare --port 8000
  python setup_tunnel.py --type tailscale
  python setup_tunnel.py --compare
        """
    )
    
    parser.add_argument(
        "--type", 
        choices=['ngrok', 'cloudflare', 'tailscale', 'localtunnel'],
        default='ngrok',
        help="Tunnel type (default: ngrok)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Local port to tunnel (default: 8000)"
    )
    parser.add_argument(
        "--token", 
        type=str,
        help="Ngrok auth token (optional)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show tunnel comparison table"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        print_comparison()
        return
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              FEDERATED LEARNING - TUNNEL SETUP                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    success = False
    url = ""
    
    if args.type == 'ngrok':
        success, url = setup_ngrok(args.port, args.token)
    elif args.type == 'cloudflare':
        success, url = setup_cloudflare(args.port)
    elif args.type == 'tailscale':
        success, url = setup_tailscale()
    elif args.type == 'localtunnel':
        success, url = setup_localtunnel(args.port)
    
    if success:
        print("\n✅ Tunnel setup complete!")
        print(f"   Clients can now connect to your FL server")
        
        # Save tunnel info
        tunnel_info = {
            'type': args.type,
            'local_port': args.port,
            'public_url': url,
            'timestamp': time.time()
        }
        
        with open('tunnel_info.json', 'w') as f:
            json.dump(tunnel_info, f, indent=2)
        
        print(f"   Tunnel info saved to: tunnel_info.json")
        
        # Keep running
        print("\n   Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n   Stopping tunnel...")
    else:
        print("\n❌ Tunnel setup failed.")
        print("   See error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
