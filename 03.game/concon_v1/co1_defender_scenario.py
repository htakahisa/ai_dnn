"""Fixed defender posts and setup terrain shared by training and inference."""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import hashlib

import numpy as np

from concon_v1.co1_guard_scenarios import build_scenario
from concon_v1.co1_map_defender import MAZE_STR
from map_data_defender_setup import get_setup_mask


@dataclass(frozen=True)
class DefenderScenario:
    grid: np.ndarray
    setup_grid: np.ndarray
    positions: dict
    setup_positions: dict
    facing_points: dict
    signature: str

    @property
    def save_dir(self):
        return Path(__file__).resolve().parent / "data" / "defender_search_data"

    @property
    def model_path(self):
        return self.save_dir / "co1_defender_search_positioning.pt"

    def battle_model_path(self, suffix="best"):
        return self.save_dir / f"co1_defender_search_{suffix}.pt"

    @property
    def runtime_model_path(self):
        best = self.battle_model_path()
        if best.is_file():
            import torch
            checkpoint = torch.load(best, map_location="cpu", weights_only=False)
            try:
                validate_checkpoint(checkpoint, self)
            except ValueError:
                return self.model_path
            return best
        return self.model_path


@lru_cache(maxsize=1)
def get_scenario():
    parsed = build_scenario("defender", MAZE_STR, "left")
    setup = parsed.grid.copy()
    mask = np.asarray(get_setup_mask())
    if mask.shape != setup.shape:
        raise ValueError("defender setup dimensions do not match terrain")
    setup[mask == 1] = 1
    from concon_v1.co1_attacker_common import bfs_distance_map
    spawns = list(zip(*np.where(parsed.grid == 4)))
    reachable = bfs_distance_map(setup, spawns[0])
    candidates = [tuple(map(int, cell)) for cell in np.argwhere(reachable >= 0)]
    setup_positions = {}
    for letter, point in parsed.positions.items():
        distances = bfs_distance_map(parsed.grid, point)
        if any(distances[start] < 0 for start in spawns):
            raise ValueError(f"defender post {letter} cannot be reached after setup")
        available = [cell for cell in candidates if cell not in setup_positions.values()]
        setup_positions[letter] = min(available, key=lambda cell: (distances[cell], reachable[cell], cell))
    signature = hashlib.sha256((parsed.signature + repr(mask.tolist())).encode()).hexdigest()
    return DefenderScenario(parsed.grid, setup, parsed.positions, setup_positions, parsed.facing_points, signature)


def phase_scenario(scenario, setup):
    return replace(scenario, grid=scenario.setup_grid if setup else scenario.grid,
                   positions=scenario.setup_positions if setup else scenario.positions)


def validate_checkpoint(checkpoint, scenario):
    if (checkpoint.get("policy_type") not in ("concon_defender_search_positioning_v1", "concon_defender_search_v1")
            or checkpoint.get("scenario_signature") != scenario.signature):
        raise ValueError("defender positioning checkpoint/map mismatch; retrain the basic model")
