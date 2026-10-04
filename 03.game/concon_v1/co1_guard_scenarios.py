"""Postplant map registry shared by guard training, evaluation and inference."""

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
import hashlib

import numpy as np

from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR, parse_game_grid


@dataclass(frozen=True)
class GuardSettings:
    map_module: str
    plant_side: str


SCENARIOS = {
    "L": GuardSettings("co1_map_guard_L", "left"),
    "R": GuardSettings("co1_map_guard_R", "right"),
}


@dataclass(frozen=True)
class GuardScenario:
    map_name: str
    plant_side: str
    grid: np.ndarray
    positions: dict
    facing_points: dict
    plant_cells: tuple
    signature: str

    @property
    def save_dir(self):
        return Path(__file__).resolve().parent / "data" / f"guard_{self.map_name}_data"

    def checkpoint_filename(self, suffix="best"):
        return f"co1_guard_{self.map_name}_{suffix}.pt"

    @property
    def model_path(self):
        return self.save_dir / self.checkpoint_filename()


def build_scenario(map_name, text, plant_side):
    if plant_side not in ("left", "right"):
        raise ValueError("guard plant_side must be left or right")
    grid = parse_game_grid(GAME_MAZE_STR)
    rows = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(rows) != grid.shape[0] or any(len(row) != grid.shape[1] for row in rows):
        raise ValueError("guard map dimensions must match the game terrain")
    markers = {letter: [] for letter in "abcdeABCDE"}
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if value in markers:
                if grid[r, c] == 1:
                    raise ValueError(f"guard marker {value} at {(r, c)} is on a wall")
                markers[value].append((r, c))
            elif not value.isdigit() or int(value) != int(grid[r, c]):
                raise ValueError(f"guard map changes terrain at {(r, c)}")
    for letter, points in markers.items():
        if len(points) != 1:
            raise ValueError(f"guard map requires exactly one {letter}; found {len(points)}")
    positions = {letter: markers[letter][0] for letter in "abcde"}
    facing_points = {letter: markers[letter.upper()][0] for letter in "abcde"}
    from concon_v1.co1_attacker_common import bfs_distance_map
    for letter, point in positions.items():
        if point == facing_points[letter]:
            raise ValueError(f"{letter} and {letter.upper()} must be different cells")
    plants = tuple((int(r), int(c)) for r, c in zip(*np.where(grid == 2))
                   if (c < grid.shape[1] / 2) == (plant_side == "left"))
    if not plants:
        raise ValueError("guard site has no plant cells")
    for point in positions.values():
        if any(bfs_distance_map(grid, plant)[point] < 0 for plant in plants):
            raise ValueError(f"guard point {point} cannot reach its site")
    signature = hashlib.sha256(
        (GAME_MAZE_STR + "\n" + "\n".join(rows) + plant_side).encode("utf-8")
    ).hexdigest()
    return GuardScenario(map_name, plant_side, grid, positions, facing_points, plants, signature)


@lru_cache(maxsize=32)
def _load_scenario(name):
    if name not in SCENARIOS:
        raise ValueError(f"unknown guard map {name!r}; choose from {tuple(SCENARIOS)}")
    settings = SCENARIOS[name]
    module = import_module(f"concon_v1.{settings.map_module}")
    return build_scenario(name, module.MAZE_STR, settings.plant_side)


def get_scenario(name="L"):
    return name if isinstance(name, GuardScenario) else _load_scenario(name)


def validate_checkpoint(checkpoint, scenario):
    scenario = get_scenario(scenario)
    if checkpoint.get("policy_type") != "concon_guard_v1":
        raise ValueError("checkpoint is not a ConCon guard model")
    if checkpoint.get("map_name") != scenario.map_name:
        raise ValueError("guard checkpoint map does not match the requested map")
    if checkpoint.get("scenario_signature") != scenario.signature:
        raise ValueError("guard map/terrain changed; retrain this guard model")
