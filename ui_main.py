#!/usr/bin/env python3
"""
JARVIS Web UI launcher.
Usage: python ui_main.py [--port 3000]
"""

import asyncio
import sys
import os

# Ensure jarvis package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.server import start_server


if __name__ == "__main__":
    port = 3000
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])

    asyncio.run(start_server(port))
