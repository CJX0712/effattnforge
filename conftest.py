"""Pytest root fixture: make the repo root importable as a package root.

The project uses top-level packages (``core``, ``attn``, ``eval``, ``data``,
``pipeline``) with no ``src/`` layout, so we insert the repo root onto
``sys.path`` once at collection time.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
