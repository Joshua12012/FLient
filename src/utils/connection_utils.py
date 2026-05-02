"""
connection_utils.py — Automated client-server connection utilities

Handles port conflicts, retry logic, and auto-discovery for FL clients.
"""

import socket
import time
import random
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def find_available_port(start_port: int = 8080, max_port: int = 9000) -> int:
    """
    Find an available port in the given range.
    
    Args:
        start_port: Starting port number to check
        max_port: Maximum port number to check
        
    Returns:
        First available port found
        
    Raises:
        RuntimeError: If no available ports found in range
    """
    for port in range(start_port, max_port + 1):
        if is_port_available(port):
            logger.info(f"Found available port: {port}")
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{max_port}")


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def is_server_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a server is reachable at the given host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def scan_for_server(
    host: str = "127.0.0.1",
    start_port: int = 8080,
    end_port: int = 9000,
    timeout: float = 1.0
) -> Optional[int]:
    """
    Scan a range of ports to find a running FL server.
    
    Args:
        host: Host to scan
        start_port: Starting port number
        end_port: Ending port number
        timeout: Connection timeout per port
        
    Returns:
        Port number if server found, None otherwise
    """
    logger.info(f"Scanning {host}:{start_port}-{end_port} for FL server...")
    
    for port in range(start_port, end_port + 1):
        if is_server_reachable(host, port, timeout):
            logger.info(f"Found server at {host}:{port}")
            return port
    
    logger.warning(f"No server found in range {start_port}-{end_port}")
    return None


class ConnectionRetryManager:
    """
    Manages connection retries with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 1.5,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.attempt = 0
        
    def get_next_delay(self) -> float:
        """Calculate next delay with exponential backoff and optional jitter."""
        delay = min(
            self.base_delay * (self.exponential_base ** self.attempt),
            self.max_delay
        )
        
        if self.jitter:
            delay = delay * (0.5 + random.random())
            
        return delay
    
    def sleep(self):
        """Sleep for the next delay period."""
        delay = self.get_next_delay()
        logger.info(f"Retry {self.attempt + 1}/{self.max_retries}: waiting {delay:.1f}s...")
        time.sleep(delay)
        self.attempt += 1
    
    def should_retry(self) -> bool:
        """Check if we should attempt another retry."""
        return self.attempt < self.max_retries
    
    def reset(self):
        """Reset the retry counter."""
        self.attempt = 0


def resolve_server_address(
    server_address: str,
    auto_discover: bool = True,
    discover_range: Tuple[int, int] = (8080, 8100)
) -> str:
    """
    Resolve server address with auto-discovery fallback.
    
    Args:
        server_address: Original server address (host:port or just host)
        auto_discover: Whether to scan for server if direct connection fails
        discover_range: Port range to scan for auto-discovery
        
    Returns:
        Resolved server address (host:port)
    """
    # Parse the address
    if ":" in server_address:
        host, port_str = server_address.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            host = server_address
            port = None
    else:
        host = server_address
        port = None
    
    # If port specified, try it first
    if port is not None:
        if is_server_reachable(host, port):
            logger.info(f"Server reachable at {host}:{port}")
            return f"{host}:{port}"
        
        logger.warning(f"Cannot reach server at {host}:{port}")
        
        if not auto_discover:
            # Return original anyway, let caller handle failure
            return f"{host}:{port}"
    
    # Auto-discover: scan for server
    if auto_discover:
        found_port = scan_for_server(host, discover_range[0], discover_range[1])
        if found_port:
            return f"{host}:{found_port}"
    
    # Fallback: return original address
    if port is None:
        port = 8080  # Default port
    return f"{host}:{port}"


def connect_with_retry(
    connect_fn,
    server_address: str,
    max_retries: int = 10,
    auto_discover: bool = True,
    on_connecting: Optional[callable] = None
):
    """
    Connect to server with automatic retry and port discovery.
    
    Args:
        connect_fn: Function that performs the actual connection
        server_address: Target server address
        max_retries: Maximum retry attempts
        auto_discover: Whether to scan for alternative ports
        on_connecting: Optional callback(address) before each attempt
        
    Returns:
        Result from connect_fn
        
    Raises:
        ConnectionError: If all retry attempts fail
    """
    retry_mgr = ConnectionRetryManager(max_retries=max_retries)
    last_error = None
    current_address = server_address
    
    while retry_mgr.should_retry():
        # On first attempt, or periodically, try to discover server
        if retry_mgr.attempt == 0 or retry_mgr.attempt % 3 == 0:
            if auto_discover:
                resolved = resolve_server_address(current_address, auto_discover=True)
                if resolved != current_address:
                    logger.info(f"Discovered server at new address: {resolved}")
                    current_address = resolved
                    retry_mgr.reset()  # Reset retries on new discovery
        
        if on_connecting:
            on_connecting(current_address)
        
        try:
            logger.info(f"Attempting connection to {current_address}...")
            return connect_fn(current_address)
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            last_error = e
            logger.warning(f"Connection failed: {e}")
            retry_mgr.sleep()
    
    raise ConnectionError(
        f"Failed to connect after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
