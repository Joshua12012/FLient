"""
kivy_client.py — Kivy GUI for mobile federated learning client

Simplified Android app for participating in federated learning using Flower framework.
Connects to a server running on your PC for federated training.
"""

import os
import sys
import threading
import time
import numpy as np
from datetime import datetime

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty
from kivy.clock import Clock


class KivyFLApp(App):
    status_text = StringProperty('Ready')

    def build(self):
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

        # Start button
        self.fl_btn = Button(text='Start Federated Learning',
                            size_hint_y=None, height=60, font_size=16)
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

    def run_flower_client_thread(self, server, client_id):
        start_time = time.time()
        self.append_status(f'=== Starting Flower Client ===')
        self.append_status(f'Target server: {server}')
        self.append_status(f'Client ID: {client_id}')

        # Import Flower client here to avoid issues on Android
        try:
            self.append_status('[1/5] Importing TensorFlow client module...')
            import sys
            from tf_client import main as client_main

            # Set command line arguments for client
            sys.argv = [
                'tf_client.py',
                '--server', server,
                '--client_id', client_id,
                '--epochs', '1',
                '--batch_size', '32',
                '--variant', 'small',
            ]

            self.append_status('[2/5] Initializing Flower client...')
            step_start = time.time()
            self.append_status('[3/5] Loading FEMNIST dataset shard...')

            # Pass progress callback to receive epoch updates
            def progress_callback(message):
                self.append_status(message)

            client_main(progress_callback=progress_callback)

            step_duration = time.time() - step_start
            total_duration = time.time() - start_time
            self.append_status('[4/5] Training completed successfully')
            self.append_status(f'[5/5] Client finished in {total_duration:.1f}s (training: {step_duration:.1f}s)')
            self.append_status('=== Session Complete ===')

        except ImportError as e:
            error_msg = str(e)
            if 'tensorflow' in error_msg or 'tf' in error_msg:
                self.append_status('❌ TensorFlow not found on this device')
                self.append_status('This app requires TensorFlow for federated learning.')
                self.append_status('SOLUTION: Rebuild the app with buildozer after updating buildozer.spec')
                self.append_status('The buildozer.spec now includes tensorflow in requirements.')
                self.append_status('Run: buildozer android clean && buildozer android debug')
            else:
                self.append_status(f'❌ Import Error: {e}')
                self.append_status('Make sure tf_client.py and dependencies are available')
        except ConnectionRefusedError as e:
            self.append_status(f'❌ Connection Refused: {e}')
            self.append_status(f'Check if server is running at {server}')
            self.append_status('Make sure firewall allows the connection')
        except TimeoutError as e:
            self.append_status(f'❌ Connection Timeout: {e}')
            self.append_status('Server took too long to respond')
        except Exception as exc:
            self.append_status(f'❌ Error: {type(exc).__name__}: {exc}')
            import traceback
            self.append_status(f'Traceback: {traceback.format_exc()[:500]}')
        finally:
            self.fl_btn.disabled = False
            self.fl_btn.text = 'Start Federated Learning'


if __name__ == '__main__':
    KivyFLApp().run()
