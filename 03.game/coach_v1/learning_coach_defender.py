"""Defender actor entry point paired only with train_coach_defender.py."""

from __future__ import annotations

from pathlib import Path

from coach_v1.common.constants import COACH_CHECKPOINT_PATHS
from coach_v1.common.types import Side
from coach_v1.learning_coach import load_coach_policy


DEFAULT_DIRECTORY = COACH_CHECKPOINT_PATHS[Side.DEFENDER.value] / "task13"


def load_defender_coach(path: Path | None = None, *, device: str = "cpu",
                        watch_points_path: Path | None = None):
    return load_coach_policy(Side.DEFENDER, path or DEFAULT_DIRECTORY / "latest.pt",
                             device=device, watch_points_path=watch_points_path)
