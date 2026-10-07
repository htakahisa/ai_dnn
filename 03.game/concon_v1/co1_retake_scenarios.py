"""Validated utility maps and separate left/right retake checkpoints."""

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
import hashlib

import numpy as np
from concon_v1.co1_retake_config import COORDINATION_VERSION

from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR, parse_game_grid


@dataclass(frozen=True)
class RetakeScenario:
    map_name: str
    grid: np.ndarray
    points: dict
    signature: str
    rally_points: tuple = ()
    assembly_labels: tuple = ()
    entry_points: tuple = ()

    def entries_for(self, rally):
        labels = dict(self.assembly_labels)
        return dict(self.entry_points).get(labels.get(rally, "").upper(), ())

    @property
    def rally_groups(self):
        """Connected cells with the same assembly label form one group."""
        remaining, groups = set(self.rally_points), []
        labels = dict(self.assembly_labels)
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            group, pending = {seed}, [seed]
            while pending:
                r, c = pending.pop()
                neighbors = {(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
                for point in sorted(neighbors & remaining):
                    if labels.get(point) != labels.get(seed):
                        continue
                    remaining.remove(point)
                    group.add(point)
                    pending.append(point)
            groups.append(tuple(sorted(group)))
        return tuple(groups)

    @property
    def save_dir(self):
        return Path(__file__).resolve().parent / "data" / f"defender_retake_{self.map_name}_data"

    def model_path(self, suffix="best"):
        return self.save_dir / f"co1_defender_retake_{self.map_name}_{suffix}.pt"


def build_scenario(name, text):
    if name not in ("L", "R"):
        raise ValueError("retake site must be L or R")
    grid = parse_game_grid(GAME_MAZE_STR)
    rows = [row.strip() for row in text.strip().splitlines() if row.strip()]
    if len(rows) != grid.shape[0] or any(len(row) != grid.shape[1] for row in rows):
        raise ValueError("retake map dimensions differ from game terrain")
    points = {ability: [] for ability in ("SMOKE", "FLASH", "RECON")}
    markers = dict(S="SMOKE", F="FLASH", R="RECON")
    rally, labels = [], []
    paired = any(value in text for value in ("a", "b"))
    entries = {label: [] for label in "AB"}
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if value in markers:
                if grid[r, c] == 1:
                    raise ValueError(f"retake marker on wall: {(r, c)}")
                points[markers[value]].append((r, c))
            elif value in ("a", "b", "A", "B"):
                # Shared map annotation, not a utility target.
                if grid[r, c] == 1:
                    raise ValueError(f"{value} marker is on a wall")
                if paired and value in "AB":
                    entries[value].append((r, c))
                elif value in "ab" or (not paired and value == "A"):
                    rally.append((r, c))
                    labels.append(((r, c), value))
            elif not value.isdigit() or int(value) != grid[r, c]:
                raise ValueError(f"retake map changes terrain at {(r, c)}")
    if any(not targets for targets in points.values()):
        raise ValueError("retake map requires S, F and R targets")
    signature = hashlib.sha256((GAME_MAZE_STR + name + '\n'.join(rows)).encode()).hexdigest()
    if not rally:
        raise ValueError("retake map requires assembly points")
    if paired and (set(dict(labels).values()) != set("ab") or any(not group for group in entries.values())):
        raise ValueError("retake map requires paired a/A and b/B markers")
    return RetakeScenario(name, grid, {key: tuple(value) for key, value in points.items()}, signature,
                          tuple(rally), tuple(labels), tuple((key, tuple(value)) for key, value in entries.items()) if paired else ())


@lru_cache(maxsize=2)
def get_scenario(name="L"):
    module = import_module(f"concon_v1.co1_map_retake_{name}")
    return build_scenario(name, module.MAZE_STR)


def plant_site(position, grid):
    return "L" if position[1] < grid.shape[1] / 2 else "R"


def normalize_ability_distances(value=6):
    """Accept per-ability limits or the legacy shared distance."""
    names = ("FLASH", "RECON", "SMOKE")
    distances = dict(value) if isinstance(value, dict) else {name: value for name in names}
    if set(distances) != set(names) or any(type(distance) is not int or distance < 1
                                         for distance in distances.values()):
        raise ValueError("FLASH, RECON and SMOKE BFS distances must be positive integers")
    return distances


def normalize_site_ability_distances(value=6):
    """Expand legacy shared limits, or validate independent L/R limits."""
    if isinstance(value, dict) and set(value) == {"L", "R"}:
        return {site: normalize_ability_distances(value[site]) for site in ("L", "R")}
    shared = normalize_ability_distances(value)
    return {site: dict(shared) for site in ("L", "R")}


def make_checkpoint(model, site, episode, ability_distance, search_path, opponents, samples):
    from concon_v1.co1_retake_common import ACTION_DIM, observation_dim, GORIGONS
    scenario = get_scenario(site)
    return dict(policy_type="concon_defender_retake_v1", map_name=site, scenario_signature=scenario.signature,
                obs_dim=observation_dim(scenario), n_actions=ACTION_DIM, training_roster=list(GORIGONS.players),
                ability_distances=normalize_site_ability_distances(ability_distance)[site], episode=episode, retake_transitions=samples,
                reward_version=2, coordination_version=COORDINATION_VERSION, phase_scope="real_plant_to_round_end", defender_perception="production_iq",
                foundation_version=1 if model.foundation else 0,
                foundation_evaluation=getattr(model, "foundation_evaluation", None),
                search_checkpoint=str(search_path), opponents=list(opponents),
                model_state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()})


def validate_checkpoint(checkpoint, scenario, ability_distance, *, allow_legacy_coordination=False):
    from concon_v1.co1_retake_common import ACTION_DIM, observation_dim, GORIGONS
    expected = dict(policy_type="concon_defender_retake_v1", map_name=scenario.map_name,
                    scenario_signature=scenario.signature,
                    obs_dim=observation_dim(scenario), n_actions=ACTION_DIM,
                    training_roster=list(GORIGONS.players), reward_version=2, coordination_version=COORDINATION_VERSION)
    for key, value in expected.items():
        if key == "coordination_version" and allow_legacy_coordination and checkpoint.get(key) in (1, 2):
            continue
        if checkpoint.get(key) != value:
            raise ValueError(f"retake checkpoint mismatch: {key}")
    saved = checkpoint.get("ability_distances", checkpoint.get("ability_distance", None))
    if normalize_ability_distances(saved) != normalize_ability_distances(ability_distance):
        raise ValueError("retake checkpoint mismatch: ability_distances")
