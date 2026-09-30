"""Match entry point for coach_v1 and its selectable checkpoint sets.

Add a checkpoint set to MODEL_PROFILES and change ACTIVE_MODEL to switch the
models used by both run_game.py and run_competition_manager.py. Paths are
relative to coach_v1/checkpoints; no model files are copied or renamed.
"""

from __future__ import annotations

import os
from pathlib import Path

from coach_v1.common.constants import CHARACTER_CHECKPOINT_IDS, CHECKPOINTS_DIR, FIXED_ROSTER_NAMES
from coach_v1.team_ai import build_coach_v1_team


MODEL_PROFILES = {
    "current": {
        "attacker": "coach/attacker/task12/latest.pt",
        "defender": "coach/defender/task13/latest.pt",
        "gorimaru": "characters/gorimaru/best.pt",
        "gongon": "characters/gongon/best.pt",
        "gongon_defender": "characters/gongon/defender_best.pt",
        "gonta": "characters/gonta/best.pt",
        "kunta": "characters/kunta/best.pt",
        "kurimaru": "characters/kurimaru/best.pt",
    },
}

ACTIVE_MODEL = "current"


def validate_coach_roster(roster) -> None:
    """The trained policies require these five character names on each side."""
    if len(roster) != len(FIXED_ROSTER_NAMES) or set(map(str, roster)) != set(FIXED_ROSTER_NAMES):
        raise ValueError(
            "coach_v1 requires the Gorigons roster: " + ", ".join(FIXED_ROSTER_NAMES)
        )


def build_game_team(*, profile: str | None = None, device: str = "cpu"):
    selected = profile or os.environ.get("COACH_V1_MODEL", ACTIVE_MODEL)
    try:
        model = MODEL_PROFILES[selected]
    except KeyError as exc:
        raise ValueError(f"unknown coach_v1 model profile: {selected}") from exc

    expected = {"attacker", "defender", "gongon_defender", *CHARACTER_CHECKPOINT_IDS}
    if set(model) != expected:
        raise ValueError(f"coach_v1 model profile {selected!r} must define {sorted(expected)}")

    def checkpoint(key: str) -> Path:
        path = (CHECKPOINTS_DIR / model[key]).resolve()
        if not path.is_relative_to(CHECKPOINTS_DIR.resolve()):
            raise ValueError(f"coach_v1 checkpoint must be under {CHECKPOINTS_DIR}: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"coach_v1 {selected} {key} checkpoint not found: {path}")
        return path

    paths = {key: checkpoint(key) for key in expected}
    return build_coach_v1_team(
        name="coach_v1",
        attacker_checkpoint=paths["attacker"],
        defender_checkpoint=paths["defender"],
        character_checkpoints={key: paths[key] for key in CHARACTER_CHECKPOINT_IDS},
        gongon_defender_checkpoint=paths["gongon_defender"],
        device=device,
    )
