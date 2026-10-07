"""Shared and opponent-specific retake model locations."""

from pathlib import Path

from concon_v1.co1_retake_scenarios import get_scenario

OPPONENT_NAMES = {
    "Ghost Champions v1": "gc_v1",
    "FRC v1": "frc_v1",
    "Fnatic v3": "fnatic_v3",
    "Touyama Gaming v2": "touyama_v2",
    "Omoko Gaming v1": "omoko_v1",
    "Toru AI v3.1": "toru_ai_v3.1",
}
OPPONENT_KEYS = tuple(OPPONENT_NAMES.values())
AI_DIRECTORY = Path(__file__).resolve().parent


def opponent_directory(opponent, kind="data"):
    if opponent not in OPPONENT_KEYS:
        raise ValueError(f"unknown retake opponent: {opponent}")
    return AI_DIRECTORY / kind / "defender_retake_opponents" / opponent


def opponent_model_path(site, opponent, suffix="best"):
    return opponent_directory(opponent) / get_scenario(site).model_path(suffix).name


def game_opponent(game):
    ai = getattr(game, "current_attacker_team_ai", None)
    return OPPONENT_NAMES.get(getattr(ai, "name", None))


def runtime_model_path(site, opponent=None):
    if opponent is not None:
        dedicated = opponent_model_path(site, opponent)
        if dedicated.is_file():
            return dedicated
    return get_scenario(site).model_path()
