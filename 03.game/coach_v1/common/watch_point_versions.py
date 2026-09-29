"""Versioned watch-point assets retained for reproducible historical actors."""

from pathlib import Path


LEGACY_WATCH_POINTS_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "watch_points_before_task16.json"
)
LEGACY_WATCH_POINTS_HASH = "2c5ff118249917e926f0c7c8aba206642aaa5a45a2d33b29cbfa8dcb9aee31b7"
TASK16_WATCH_POINTS_HASH = "c632c399cb359a46ee82c8c2120a74ac5367575a78730ed390110f191f06336e"
