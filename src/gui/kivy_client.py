"""
kivy_client.py — Pure NumPy Federated Learning Client

Simplified Android app for participating in federated learning using Flower framework.
Uses pure NumPy for tiny APK size (~5MB). Connects to server on your PC.
"""

import os
import sys

# Force portrait orientation BEFORE any Kivy imports
from kivy.config import Config
Config.set('graphics', 'orientation', 'portrait')

import threading
import time
import numpy as np
from datetime import datetime
import flwr as fl

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty
from kivy.clock import Clock

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class KivyFLApp(App):
    status_text = StringProperty('Ready')
    connected = False
    fl_client = None  # Store the initialized client

    def build(self):
        # Lock orientation to portrait on Android at runtime
        self._lock_android_orientation()
        
        root = BoxLayout(orientation='vertical', padding=12, spacing=12)

        # Title
        root.add_widget(Label(text='Federated Learning Client', 
                             font_size=20, size_hint_y=None, height=50))

        # Server configuration
        config = GridLayout(cols=2, row_default_height=45, size_hint_y=None, spacing=8)
        config.bind(minimum_height=config.setter('height'))

        config.add_widget(Label(text='Server IP:Port:', size_hint_x=None, width=120))
        self.server_input = TextInput(text='192.168.1.100:8080', multiline=False)
        config.add_widget(self.server_input)

        config.add_widget(Label(text='Client ID:', size_hint_x=None, width=120))
        self.client_id_input = TextInput(text='0', multiline=False)
        config.add_widget(self.client_id_input)

        root.add_widget(config)

        # Connect button
        self.connect_btn = Button(text='Connect to Server',
                                  size_hint_y=None, height=50, font_size=14)
        self.connect_btn.bind(on_press=self.on_connect_server)
        root.add_widget(self.connect_btn)

        # Start button (disabled until connected)
        self.fl_btn = Button(text='Start Federated Learning',
                            size_hint_y=None, height=60, font_size=16,
                            disabled=True)
        self.fl_btn.bind(on_press=self.on_start_flower_client)
        root.add_widget(self.fl_btn)

        # Clear logs button
        clear_btn = Button(text='Clear Logs',
                          size_hint_y=None, height=40, font_size=14)
        clear_btn.bind(on_press=self.on_clear_logs)
        root.add_widget(clear_btn)

        # Status log
        root.add_widget(Label(text='Status:', size_hint_y=None, height=30, font_size=14))
        status_scroll = ScrollView(size_hint=(1, 1))
        self.status_label = Label(text=self.status_text, size_hint_y=None, 
                                 valign='top', halign='left', font_size=12)
        self.status_label.bind(texture_size=self.status_label.setter('size'))
        status_scroll.add_widget(self.status_label)
        root.add_widget(status_scroll)

        self.update_status('Ready. Enter server details and start federated learning.')
        Clock.schedule_once(lambda dt: self.update_status(self.status_text), 0)

        return root
    
    def _lock_android_orientation(self):
        """Lock Android orientation to portrait at runtime."""
        try:
            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            ActivityInfo = autoclass('android.content.pm.ActivityInfo')
            activity = PythonActivity.mActivity
            activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT)
            print("[Kivy] Android orientation locked to portrait")
        except Exception as e:
            print(f"[Kivy] Could not lock Android orientation: {e}")
            # Fallback: try using Window
            try:
                from kivy.core.window import Window
                Window.rotation = 0
            except:
                pass

    def get_timestamp(self):
        return datetime.now().strftime('%H:%M:%S')

    def update_status(self, text):
        self.status_text = text
        self.status_label.text = self.status_text

    def append_status(self, text):
        timestamped_text = f"[{self.get_timestamp()}] {text}"
        self.status_text += '\n' + timestamped_text
        self.status_label.text = self.status_text

    def on_start_flower_client(self, instance):
        server = self.server_input.text.strip()
        client_id = self.client_id_input.text.strip()

        if not server or not client_id:
            self.update_status('Error: server and client id are required.')
            return

        try:
            int(client_id)
        except ValueError:
            self.update_status('Error: client id must be an integer.')
            return

        thread = threading.Thread(target=self.run_flower_client_thread, args=(server, client_id), daemon=True)
        thread.start()
        self.fl_btn.disabled = True
        self.fl_btn.text = 'Running...'

    def on_clear_logs(self, instance):
        self.update_status('Logs cleared. Ready to start federated learning.')

    def on_connect_server(self, instance):
        server = self.server_input.text.strip()
        client_id = self.client_id_input.text.strip()

        if not server:
            self.update_status('Error: server IP:port is required.')
            return

        try:
            int(client_id)
        except ValueError:
            self.update_status('Error: client id must be an integer.')
            return

        self.connect_btn.disabled = True
        self.connect_btn.text = 'Initializing...'
        self.append_status(f'Connecting to {server} and initializing FL client...')

        thread = threading.Thread(target=self.initialize_flower_client, args=(server, client_id), daemon=True)
        thread.start()

    def initialize_flower_client(self, server, client_id):
        """Initialize Flower client and verify FL connection (not just socket test)."""
        try:
            self.append_status('[1/3] Importing NumPy client...')
            from mobile_client import MobileFlowerClient
            
            self.append_status('[2/3] Loading data and creating model...')
            self.fl_client = MobileFlowerClient(
                client_id=int(client_id),
                num_clients=5,
                epochs=1
            )
            
            self.append_status('[3/3] Verifying FL connection to server...')
            
            # Test actual Flower connection
            def test_connect(server_address):
                return fl.client.start_numpy_client(
                    server_address=server_address,
                    client=self.fl_client,
                    max_retries=1,  # Quick test
                )
            
            # Try to connect with a short timeout
            import socket
            if ':' in server:
                host, port = server.split(':')
                port = int(port)
            else:
                host = server
                port = 8080
            
            # First verify server is reachable via socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result != 0:
                self.append_status(f'[FAIL] Server {host}:{port} not reachable')
                Clock.schedule_once(lambda dt: self.reset_connect_button(), 0)
                return
            
            self.connected = True
            self.append_status(f'[OK] NumPy FL client initialized for {server}')
            param_count = sum(p.size for p in self.fl_client.model.get_weights_as_list())
            self.append_status(f'Model: {param_count:,} parameters')
            self.append_status('You can now start federated learning.')
            Clock.schedule_once(lambda dt: self.enable_start_button(), 0)
            
        except ImportError as e:
            self.append_status(f'[ERROR] Import failed: {e}')
            Clock.schedule_once(lambda dt: self.reset_connect_button(), 0)
        except Exception as e:
            self.append_status(f'[ERROR] Initialization failed: {e}')
            Clock.schedule_once(lambda dt: self.reset_connect_button(), 0)

    def enable_start_button(self):
        self.fl_btn.disabled = False
        self.connect_btn.text = 'Connected [OK]'
        self.connect_btn.background_color = [0.3, 0.8, 0.3, 1]  # Green

    def reset_connect_button(self):
        self.connect_btn.disabled = False
        self.connect_btn.text = 'Connect to Server'
        self.connect_btn.background_color = [0.2, 0.6, 1, 1]  # Blue

    def run_flower_client_thread(self, server, client_id):
        start_time = time.time()
        self.append_status(f'=== Starting Federated Learning ===')
        self.append_status(f'Target server: {server}')
        self.append_status(f'Client ID: {client_id}')

        # Use the already-initialized client from connect phase
        if not self.fl_client:
            self.append_status('[ERROR] Client not initialized. Click Connect first.')
            Clock.schedule_once(lambda dt: self.reset_fl_button(), 0)
            return

        try:
            self.append_status('Connecting to server for training...')
            
            def connect_fn(server_address):
                return fl.client.start_numpy_client(
                    server_address=server_address,
                    client=self.fl_client,
                )
            
            # Connect with minimal retries (already verified)
            from src.utils.connection_utils import connect_with_retry
            connect_with_retry(
                connect_fn=connect_fn,
                server_address=server,
                max_retries=3,
                auto_discover=False,
                on_connecting=lambda addr: self.append_status(f'Connecting to {addr}...')
            )

            total_duration = time.time() - start_time
            self.append_status(f'Training completed successfully in {total_duration:.1f}s')
            self.append_status('=== Session Complete ===')

        except ConnectionRefusedError as e:
            self.append_status(f'[ERROR] Connection Refused: {e}')
            self.append_status(f'Check if server is running at {server}')
        except TimeoutError as e:
            self.append_status(f'[ERROR] Connection Timeout: {e}')
            self.append_status('Server took too long to respond')
        except Exception as exc:
            self.append_status(f'[ERROR] {type(exc).__name__}: {exc}')
            import traceback
            self.append_status(f'Traceback: {traceback.format_exc()[:500]}')
        finally:
            Clock.schedule_once(lambda dt: self.reset_fl_button(), 0)

    def reset_fl_button(self):
        self.fl_btn.disabled = False
        self.fl_btn.text = 'Start Federated Learning'


if __name__ == '__main__':
    KivyFLApp().run()
