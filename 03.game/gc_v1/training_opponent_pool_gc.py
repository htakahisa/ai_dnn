"""Actual rosters and dedicated AIs used by GC real-engine training.

Use opponent_for_episode for an even, reproducible five-team rotation. Toru
v3's engine key is v3.1 (the GUI name); Fnatic v3 is its dedicated rule AI.
"""
from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
import random

from party_presets import get_preset

GC_PRESET_NAME = "Ghost Champions"
OPPONENT_SPECS = (
    ("Furina Classic", "frc_v1"),
    ("Touyama Gaming", "touyama_gaming_v2"),
    ("Omoko Gaming", "omoko_gaming_v1"),
    ("Fnatic2023", "fnatic_v3"),
    ("SUPES", "toru_ai_v3.1"),
)


@dataclass(frozen=True)
class TrainingOpponent:
    name: str
    players: tuple[str, ...]
    igl: str
    spike_holder: str
    ai_key: str

    def build_team_ai(self):
        from run_game import _build_team_ai
        return _build_team_ai(self.ai_key)

    def build_loaded_team_ai(self, side):
        """Legacy loaders print and swallow errors; never train against that fallback."""
        team = self.build_team_ai()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            if side == "A":
                team.get_attacker_controller()
            else:
                team.get_defender_controller()
        errors = [line for line in log.getvalue().splitlines()
                  if "LOAD ERROR" in line or "load failed" in line or "model missing" in line]
        if errors:
            raise RuntimeError(f"Dedicated opponent failed to load ({self.name}): " + " | ".join(errors))
        return team


def rotation_opponents() -> tuple[TrainingOpponent, ...]:
    opponents = []
    for name, key in OPPONENT_SPECS:
        preset = get_preset(name)
        if (preset is None or len(preset.players) != 5
                or preset.igl not in preset.players
                or preset.spike_holder not in preset.players):
            raise RuntimeError(f"Invalid training opponent preset: {name}")
        opponents.append(TrainingOpponent(
            preset.name, tuple(preset.players), preset.igl, preset.spike_holder, key))
    return tuple(opponents)


def opponent_for_episode(index: int) -> TrainingOpponent:
    opponents = rotation_opponents()
    return opponents[index % len(opponents)]


def real_team_names(*, exclude_gc: bool = True) -> list[str]:
    names = [opponent.name for opponent in rotation_opponents()]
    return names if exclude_gc else names + [GC_PRESET_NAME]


def choose_real_team_opponent(rng: random.Random, *, ai_key: str | None = None,
                              episode_index: int | None = None) -> TrainingOpponent:
    opponent = (opponent_for_episode(episode_index) if episode_index is not None
                else rng.choice(rotation_opponents()))
    if ai_key is None:
        return opponent
    return TrainingOpponent(opponent.name, opponent.players, opponent.igl,
                            opponent.spike_holder, ai_key)


def describe_pool() -> str:
    return "rotation: " + " -> ".join(
        f"{o.name} ({o.ai_key})" for o in rotation_opponents())


class OpponentRotation:
    """Reuse loaded controllers, but reset their episode memory and IQ cache."""

    def __init__(self):
        self.teams = {}
        self.current = None

    def bind(self, game, episode_index: int, *, side="D"):
        from run_competition_manager import TeamPlayerKey
        opponent = opponent_for_episode(episode_index)
        if opponent.ai_key not in self.teams:
            self.teams[opponent.ai_key] = opponent.build_loaded_team_ai(side)
        team = self.teams[opponent.ai_key]
        prefix = "defender" if side == "D" else "attacker"
        setattr(game, prefix + "_roster", [
            TeamPlayerKey(name, "opponent:" + opponent.name) for name in opponent.players])
        setattr(game, prefix + "_igl_name", opponent.igl)
        setattr(game, prefix + "_team_name", opponent.name)
        if side == "D":
            game.defender_spike_holder_name = opponent.spike_holder
            game.initial_defender_team_ai = team
            game.current_defender_team_ai = team
        else:
            game.spike_holder_name = opponent.spike_holder
            game.initial_attacker_team_ai = team
            game.current_attacker_team_ai = team
        game._refresh_active_controllers()
        team.reset_round()
        self.current = opponent
        return opponent
