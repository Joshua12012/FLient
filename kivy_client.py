"""
kivy_client.py — Kivy GUI for mobile federated learning client

Simplified Android app for participating in federated learning using Flower framework.
Connects to a server running on your PC for federated training.
"""

import os
import sys
import threading
import numpy as np

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

        config.add_widget(Label(text='Total clients:', size_hint_x=None, width=120))
        self.num_clients_input = TextInput(text='5', multiline=False)
        config.add_widget(self.num_clients_input)

        root.add_widget(config)

        # Start button
        self.fl_btn = Button(text='Start Federated Learning', 
                            size_hint_y=None, height=60, font_size=16)
        self.fl_btn.bind(on_press=self.on_start_flower_client)
        root.add_widget(self.fl_btn)

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

    def update_status(self, text):
        self.status_text = text
        self.status_label.text = self.status_text

    def append_status(self, text):
        self.status_text += '\n' + text
        self.status_label.text = self.status_text

    def on_start_flower_client(self, instance):
        server = self.server_input.text.strip()
        client_id = self.client_id_input.text.strip()
        num_clients = self.num_clients_input.text.strip()

        if not server or not client_id or not num_clients:
            self.update_status('Error: server, client id, and total clients are required.')
            return

        try:
            int(client_id)
            int(num_clients)
        except ValueError:
            self.update_status('Error: client id and total clients must be integers.')
            return

        thread = threading.Thread(target=self.run_flower_client_thread, args=(server, client_id, num_clients), daemon=True)
        thread.start()
        self.fl_btn.disabled = True
        self.fl_btn.text = 'Running...'

    def run_flower_client_thread(self, server, client_id, num_clients):
        self.append_status(f'Starting Flower client {client_id} → {server}')
        
        # Import Flower client here to avoid issues on Android
        try:
            from client import main as client_main
            import sys
            
            # Set command line arguments for client
            sys.argv = [
                'client.py',
                '--server', server,
                '--client_id', client_id,
                '--num_clients', num_clients,
                '--variant', 'small',
                '--epochs', '1',
                '--alpha', '0.5',
            ]
            
            self.append_status('Initializing Flower client...')
            client_main()
            self.append_status('Flower client finished successfully.')
            
        except ImportError as e:
            self.append_status(f'Error importing client module: {e}')
            self.append_status('Make sure client.py is included in the app package.')
        except Exception as exc:
            self.append_status(f'Flower client failed: {exc}')
        finally:
            self.fl_btn.disabled = False
            self.fl_btn.text = 'Start Federated Learning'


if __name__ == '__main__':
    KivyFLApp().run()
