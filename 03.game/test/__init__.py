"""Regression tests and shared fixtures for the game."""

from pathlib import Path
import sys

# Keep the existing test_* fixture imports working with package discovery, too.
_test_directory = str(Path(__file__).resolve().parent)
if _test_directory not in sys.path:
    sys.path.insert(0, _test_directory)
