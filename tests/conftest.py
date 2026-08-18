"""Pytest configuration: make the repo root importable so `src.*` resolves.

All tests live under `tests/`, which is one level below the repo root, so this
file adds the parent directory to ``sys.path``.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
