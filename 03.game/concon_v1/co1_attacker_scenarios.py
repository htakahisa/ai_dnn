"""Attack route settings shared by training, evaluation, and inference."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import import_module
from pathlib import Path

import numpy as np

from map_data import NEW_MAZE_STR as GAME_MAZE_STR


WAYPOINT_ORDER = "abcd"


@dataclass(frozen=True)
class ScenarioSettings:
    map_module: str
    plant_side: str
    waypoint_order: str = WAYPOINT_ORDER
    max_candidate_bfs_distance: int = 12
    smoke_trigger_bfs_distance: int = 6
    flash_trigger_bfs_distance: int = 6


SCENARIOS = {
    "A1": ScenarioSettings(
        map_module="co1_map_attacker_A1", plant_side="left", waypoint_order="abcd",
        smoke_trigger_bfs_distance=6,
        flash_trigger_bfs_distance=6,
    ),
    "A2": ScenarioSettings(
        map_module="co1_map_attacker_A2", plant_side="right", waypoint_order="abcde",
        max_candidate_bfs_distance=33,
        smoke_trigger_bfs_distance=8,
        flash_trigger_bfs_distance=6,
    ),
    "A3": ScenarioSettings(
        map_module="co1_map_attacker_A3", plant_side="right", waypoint_order="abcde",
        max_candidate_bfs_distance=25,
        smoke_trigger_bfs_distance=11,
        flash_trigger_bfs_distance=5,
    ),
}


def _rows(map_text):
    rows = [line.strip() for line in map_text.strip().splitlines() if line.strip()]
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("map rows must be non-empty and have equal width")
    return rows


def parse_game_grid(map_text):
    rows = _rows(map_text)
    if any(not row.isdigit() for row in rows):
        raise ValueError("game terrain map must contain digits only")
    return np.asarray([[int(value) for value in row] for row in rows], dtype=np.int32)


def parse_strategy_points(map_text, waypoint_order=WAYPOINT_ORDER):
    waypoint_order = waypoint_order.lower()
    if "s" in waypoint_order:
        raise ValueError("S is reserved for fixed smoke points")
    if "u" in waypoint_order:
        raise ValueError("U is reserved for fixed flash points")
    if (not waypoint_order or waypoint_order[0] != "a"
            or len(set(waypoint_order)) != len(waypoint_order)
            or any(marker not in "abcdefghijklmnopqrstuvwxyz" for marker in waypoint_order)):
        raise ValueError("waypoint order must contain unique letters and start with a/A")
    points = {marker: [] for marker in waypoint_order}
    marker_cases = {}
    for row_index, row in enumerate(_rows(map_text)):
        for col_index, value in enumerate(row):
            marker = value.lower()
            if value in ("S", "U"):
                continue
            if marker in points:
                if marker in marker_cases and marker_cases[marker] != value.isupper():
                    raise ValueError(f"waypoint {marker!r} must not mix uppercase and lowercase markers")
                marker_cases[marker] = value.isupper()
                points[marker].append((row_index, col_index))
            elif not value.isdigit():
                raise ValueError(f"unsupported strategy-map character: {value!r}")
    if any(not points[marker] for marker in waypoint_order):
        raise ValueError(f"strategy map must contain every waypoint in {waypoint_order!r}")
    return points


@dataclass(frozen=True, eq=False)
class AttackerScenario:
    map_name: str
    strategy_map: str
    game_map: str
    plant_side: str
    grid: np.ndarray
    waypoint_points: dict
    plant_cells: list
    attacker_spawns: list
    max_candidate_bfs_distance: int
    waypoint_order: str
    signature: str
    uppercase_markers: frozenset = frozenset()
    smoke_points: tuple = ()
    smoke_trigger_bfs_distance: int = 6
    flash_points: tuple = ()
    flash_trigger_bfs_distance: int = 6

    @property
    def obs_dim(self):
        # 8 actor/split features + waypoint/plant stages + 15 goal/status features.
        return 24 + len(self.waypoint_order)

    @property
    def save_dir(self):
        return Path(__file__).resolve().parent / "data" / f"attacker_{self.map_name}_data"

    def checkpoint_filename(self, kind="best"):
        if kind not in ("best", "latest"):
            raise ValueError("checkpoint kind must be best or latest")
        return f"co1_attacker_{self.map_name}_{kind}.pt"

    @property
    def model_path(self):
        return self.save_dir / self.checkpoint_filename()


def build_scenario(map_name, strategy_map, plant_side, game_map=GAME_MAZE_STR,
                   max_candidate_bfs_distance=12, waypoint_order=WAYPOINT_ORDER,
                   smoke_trigger_bfs_distance=6, flash_trigger_bfs_distance=6):
    waypoint_order = waypoint_order.lower()
    if (isinstance(smoke_trigger_bfs_distance, bool)
            or not isinstance(smoke_trigger_bfs_distance, int)
            or smoke_trigger_bfs_distance < 0):
        raise ValueError("smoke trigger BFS distance must be a non-negative integer")
    if (isinstance(flash_trigger_bfs_distance, bool)
            or not isinstance(flash_trigger_bfs_distance, int)
            or flash_trigger_bfs_distance < 0):
        raise ValueError("flash trigger BFS distance must be a non-negative integer")
    grid = parse_game_grid(game_map)
    points = parse_strategy_points(strategy_map, waypoint_order)
    if grid.shape != (len(_rows(strategy_map)), len(_rows(strategy_map)[0])):
        raise ValueError("strategy map dimensions must match the game terrain map")
    if any(grid[row, col] == 1 for cells in points.values() for row, col in cells):
        raise ValueError("a strategy waypoint overlays a wall in the game terrain map")
    smoke_points = tuple((r, c) for r, row in enumerate(_rows(strategy_map))
                         for c, value in enumerate(row) if value == "S")
    if any(grid[point] == 1 for point in smoke_points):
        raise ValueError("a fixed smoke point overlays a wall in the game terrain map")
    flash_points = tuple((r, c) for r, row in enumerate(_rows(strategy_map))
                         for c, value in enumerate(row) if value == "U")
    if any(grid[point] == 1 for point in flash_points):
        raise ValueError("a fixed flash point overlays a wall in the game terrain map")
    if len(points["a"]) not in (1, 2):
        raise ValueError("the first waypoint a must have one or two points")
    if any(len(cells) > 5 for cells in points.values()):
        raise ValueError("the observation supports at most five points per marker")
    if plant_side not in ("left", "right"):
        raise ValueError("plant side must be left or right")
    if max_candidate_bfs_distance < 1:
        raise ValueError("waypoint candidate BFS distance limit must be positive")
    height, width = grid.shape
    plant_cells = [(r, c) for r in range(height) for c in range(width)
                   if grid[r, c] == 2 and (c < width // 2) == (plant_side == "left")]
    spawns = [(r, c) for r in range(height) for c in range(width) if grid[r, c] == 3]
    if not plant_cells or len(spawns) != 5:
        raise ValueError("a route requires target plant cells and exactly five attacker spawns")
    contents = "\n".join(_rows(game_map) + _rows(strategy_map)
                         + [plant_side, str(max_candidate_bfs_distance)])
    # Preserve signatures of existing abcd checkpoints from the first map refactor.
    if waypoint_order != WAYPOINT_ORDER:
        contents += "\n" + waypoint_order
    if smoke_points:
        contents += "\nsmoke_trigger_bfs_distance=" + str(smoke_trigger_bfs_distance)
    if flash_points:
        contents += "\nflash_trigger_bfs_distance=" + str(flash_trigger_bfs_distance)
    signature = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    uppercase_markers = frozenset(value.lower() for row in _rows(strategy_map)
                                  for value in row if value.isupper() and value not in ("S", "U"))
    return AttackerScenario(map_name, strategy_map, game_map, plant_side,
                            grid, points, plant_cells, spawns,
                            max_candidate_bfs_distance, waypoint_order, signature, uppercase_markers,
                            smoke_points, smoke_trigger_bfs_distance,
                            flash_points, flash_trigger_bfs_distance)


@lru_cache(maxsize=None)
def _load_scenario(map_name):
    if map_name not in SCENARIOS:
        raise ValueError(f"unknown map: {map_name!r}; choose from {', '.join(SCENARIOS)}")
    settings = SCENARIOS[map_name]
    module = import_module(f"concon_v1.{settings.map_module}")
    return build_scenario(map_name, module.MAZE_STR, settings.plant_side,
                          max_candidate_bfs_distance=settings.max_candidate_bfs_distance,
                          waypoint_order=settings.waypoint_order,
                          smoke_trigger_bfs_distance=settings.smoke_trigger_bfs_distance,
                          flash_trigger_bfs_distance=settings.flash_trigger_bfs_distance)


def get_scenario(map_name="A1"):
    return map_name if isinstance(map_name, AttackerScenario) else _load_scenario(map_name)


def validate_checkpoint_scenario(checkpoint, scenario):
    """Legacy checkpoints without a map name belong to A1."""
    scenario = get_scenario(scenario)
    saved_map = checkpoint.get("map_name", "A1")
    if saved_map != scenario.map_name:
        raise ValueError(f"checkpoint map {saved_map!r} does not match requested map {scenario.map_name!r}")
    if checkpoint.get("waypoint_order", WAYPOINT_ORDER) != scenario.waypoint_order:
        raise ValueError("checkpoint waypoint order does not match this map; retrain the model")
    if checkpoint.get("obs_dim", scenario.obs_dim) != scenario.obs_dim:
        raise ValueError("checkpoint observation dimensions do not match this map; retrain the model")
    signature = checkpoint.get("scenario_signature")
    if signature is not None and signature != scenario.signature:
        raise ValueError("checkpoint route/terrain settings do not match this map")
    points = checkpoint.get("waypoint_points")
    if points is not None:
        points = {marker: [tuple(point) for point in cells] for marker, cells in points.items()}
        if points != scenario.waypoint_points:
            raise ValueError("checkpoint waypoints do not match this map")
    plants = checkpoint.get("plant_cells")
    if plants is None and saved_map == "A1":
        plants = checkpoint.get("left_plant_cells")
    if plants is not None and [tuple(point) for point in plants] != scenario.plant_cells:
        raise ValueError("checkpoint plant cells do not match this map")
