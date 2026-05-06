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
from kivy.core.window import Window
from kivy.utils import get_color_from_hex


class KivyFLApp(App):
    status_text = StringProperty('Ready')

    def build(self):
        # Set background color
        Window.clearcolor = get_color_from_hex('#1a1a2e')
        
        root = BoxLayout(orientation='vertical', padding=20, spacing=15)

        # Header with gradient-like effect
        header = BoxLayout(orientation='vertical', size_hint_y=None, height=80)
        title = Label(
            text='🌸 Federated Learning',
            font_size=28,
            font_name='Roboto',
            color=get_color_from_hex('#ffffff'),
            size_hint_y=None,
            height=50
        )
        subtitle = Label(
            text='Edge Client for Mobile Devices',
            font_size=14,
            color=get_color_from_hex('#a0a0a0'),
            size_hint_y=None,
            height=30
        )
        header.add_widget(title)
        header.add_widget(subtitle)
        root.add_widget(header)

        # Server configuration card
        config_container = BoxLayout(orientation='vertical', size_hint_y=None, height=200)
        config_container.padding = 15
        config_container.spacing = 10
        
        config_label = Label(
            text='⚙️  Server Configuration',
            font_size=18,
            color=get_color_from_hex('#ffffff'),
            size_hint_y=None,
            height=35,
            halign='left'
        )
        config_container.add_widget(config_label)

        config = GridLayout(cols=2, row_default_height=50, size_hint_y=None, spacing=10)
        config.bind(minimum_height=config.setter('height'))

        # Server IP
        config.add_widget(Label(text='Server IP:', size_hint_x=None, width=100, 
                               color=get_color_from_hex('#e0e0e0'), halign='right'))
        self.server_input = TextInput(
            text='192.168.1.100:8080', 
            multiline=False,
            size_hint_x=1,
            background_color=get_color_from_hex('#2d2d44'),
            foreground_color=get_color_from_hex('#ffffff'),
            cursor_color=get_color_from_hex('#7b68ee')
        )
        config.add_widget(self.server_input)

        # Client ID
        config.add_widget(Label(text='Client ID:', size_hint_x=None, width=100,
                               color=get_color_from_hex('#e0e0e0'), halign='right'))
        self.client_id_input = TextInput(
            text='0', 
            multiline=False,
            size_hint_x=1,
            background_color=get_color_from_hex('#2d2d44'),
            foreground_color=get_color_from_hex('#ffffff'),
            cursor_color=get_color_from_hex('#7b68ee')
        )
        config.add_widget(self.client_id_input)

        # Total clients
        config.add_widget(Label(text='Total Clients:', size_hint_x=None, width=100,
                               color=get_color_from_hex('#e0e0e0'), halign='right'))
        self.num_clients_input = TextInput(
            text='5', 
            multiline=False,
            size_hint_x=1,
            background_color=get_color_from_hex('#2d2d44'),
            foreground_color=get_color_from_hex('#ffffff'),
            cursor_color=get_color_from_hex('#7b68ee')
        )
        config.add_widget(self.num_clients_input)

        config_container.add_widget(config)
        root.add_widget(config_container)

        # Start button with modern styling
        self.fl_btn = Button(
            text='▶ Start Training',
            size_hint_y=None, 
            height=65, 
            font_size=18,
            background_color=get_color_from_hex('#7b68ee'),
            color=get_color_from_hex('#ffffff'),
            bold=True
        )
        self.fl_btn.bind(on_press=self.on_start_flower_client)
        root.add_widget(self.fl_btn)

        # Status section
        status_header = Label(
            text='📊 Training Status',
            font_size=16,
            color=get_color_from_hex('#ffffff'),
            size_hint_y=None,
            height=30,
            halign='left'
        )
        root.add_widget(status_header)

        status_scroll = ScrollView(size_hint=(1, 1), bar_color=get_color_from_hex('#7b68ee'))
        self.status_label = Label(
            text=self.status_text, 
            size_hint_y=None, 
            valign='top', 
            halign='left', 
            font_size=13,
            color=get_color_from_hex('#e0e0e0'),
            text_size=(None, None)
        )
        self.status_label.bind(texture_size=self.status_label.setter('size'))
        status_scroll.add_widget(self.status_label)
        root.add_widget(status_scroll)

        self.update_status('✓ Ready to connect\nEnter your server IP and client ID above.')
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
            from client_tf import main as client_main
            import sys
            
            # Set command line arguments for client
            sys.argv = [
                'client_tf.py',
                '--server', server,
                '--client_id', client_id,
                '--num_clients', num_clients,
                '--variant', 'small',
                '--epochs', '1',
                '--alpha', '0.5',
            ]
            
            self.append_status('Initializing Flower client (TensorFlow)...')
            client_main()
            self.append_status('Flower client finished successfully.')
            
        except ImportError as e:
            self.append_status(f'Error importing client module: {e}')
            self.append_status('Make sure client_tf.py is included in the app package.')
        except Exception as exc:
            self.append_status(f'Flower client failed: {exc}')
        finally:
            self.fl_btn.disabled = False
            self.fl_btn.text = 'Start Federated Learning'


if __name__ == '__main__':
    KivyFLApp().run()
