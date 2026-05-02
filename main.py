"""
main.py - Entry point for Android Kivy application
This file is required by Buildozer for Android packaging
"""
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gui.kivy_client import KivyFLApp

if __name__ == '__main__':
    KivyFLApp().run()
