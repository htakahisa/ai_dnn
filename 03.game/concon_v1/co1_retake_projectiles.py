"""Static fixed-point aiming using the production projectile geometry."""

from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from abilities_los import AbilityLosMixin
from grid_lines import line_cells
from concon_v1.co1_attacker_abilities import _impact_aim


def projectile_aim(grid, source, point, ability):
    return _aim(grid.shape, grid.astype(np.int32).tobytes(), tuple(source), tuple(point), ability)


@lru_cache(maxsize=2048)
def _aim(shape, terrain, source, point, ability):
    grid = np.frombuffer(terrain, dtype=np.int32).reshape(shape)
    game = SimpleNamespace(grid=grid, height=shape[0], width=shape[1], _line_cells=line_cells)
    game._projectile_path = lambda start, goal: AbilityLosMixin._projectile_path(game, start, goal)
    # target is a direction in the engine. Find an aim whose wall/edge endpoint
    # (or flash flight deadline) actually lands on the marked effect cell.
    return _impact_aim(game, SimpleNamespace(pos=source), point, flash=ability == "FLASH")
