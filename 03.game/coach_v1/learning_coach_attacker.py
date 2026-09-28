"""Attacker actor entry point paired with train_coach_attacker.py."""

from __future__ import annotations

from pathlib import Path

from coach_v1.common.constants import COACH_CHECKPOINT_PATHS
from coach_v1.common.types import Side
from coach_v1.learning_coach import load_coach_policy


DEFAULT_DIRECTORY = COACH_CHECKPOINT_PATHS[Side.ATTACKER.value] / "task12"


def load_attacker_coach(path: Path | None = None, *, device: str = "cpu"):
    return load_coach_policy(Side.ATTACKER, path or DEFAULT_DIRECTORY / "latest.pt",
                             device=device)
