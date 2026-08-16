"""Shared real-team opponent pool for Ghost Champions training.

Every training match can draw one actual team preset from party_presets.py.
Ghost Champions itself is excluded.

The opponent keeps:
- its exact 5-player roster
- its IGL
- its attacker spike holder
- all normal player-combo / awakening behavior supplied by the game

AI:
- Prefer the current competition key "toru_ai_v3.1" when run_game supports it.
- Fall back to "toru_ai_v3" for run_game versions where v3.1 is only a
  Competition Manager display alias.

This module is intentionally independent of any GC training phase so Setup,
Opening, Search and Retake can all share the same opponent distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from party_presets import all_preset_names, get_preset
from run_game import _build_team_ai


GC_PRESET_NAME = "Ghost Champions"


@dataclass(frozen=True)
class TrainingOpponent:
    name: str
    players: tuple[str, ...]
    igl: str
    spike_holder: str
    ai_key: str

    def build_team_ai(self):
        return _build_team_ai(self.ai_key)


def _resolve_toru_team_ai_key() -> str:
    """Return the strongest/current Toru team-AI key supported by this run_game."""
    errors = []
    for key in ("toru_ai_v3.1", "toru_ai_v3"):
        try:
            # Probe only. The returned object is discarded.
            _build_team_ai(key)
            return key
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No supported Toru AI team key was found in run_game._build_team_ai. "
        + " | ".join(errors)
    )


def real_team_names(*, exclude_gc: bool = True) -> list[str]:
    names = []
    for name in all_preset_names():
        if exclude_gc and name == GC_PRESET_NAME:
            continue

        preset = get_preset(name)
        if preset is None:
            continue
        if len(tuple(preset.players)) != 5:
            continue
        if preset.igl not in preset.players:
            continue
        if preset.spike_holder is None or preset.spike_holder not in preset.players:
            continue

        names.append(name)

    if not names:
        raise RuntimeError(
            "No usable real-team presets were found in party_presets.py"
        )

    return names


def choose_real_team_opponent(
    rng: random.Random,
    *,
    ai_key: str | None = None,
) -> TrainingOpponent:
    names = real_team_names(exclude_gc=True)
    name = rng.choice(names)
    preset = get_preset(name)

    if preset is None:
        raise RuntimeError(f"Preset disappeared after selection: {name}")

    resolved_ai = ai_key or _resolve_toru_team_ai_key()

    return TrainingOpponent(
        name=preset.name,
        players=tuple(preset.players),
        igl=preset.igl,
        spike_holder=preset.spike_holder,
        ai_key=resolved_ai,
    )


def describe_pool() -> str:
    names = real_team_names(exclude_gc=True)
    ai_key = _resolve_toru_team_ai_key()
    return (
        f"real teams={len(names)} / ai={ai_key} / "
        + ", ".join(names)
    )
