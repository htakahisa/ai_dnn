"""Roster-based final attack destination, independent of opponent AI/name."""

import numpy as np

from party_presets import get_preset
from .map_data_macro_gc import MACRO_ZONE_STR, ZONE_MARKERS, parse_layer
from .positioning_gc import set_team_plant_target, team_plant_target

TOUYAMA_ROSTER = frozenset(map(str, get_preset("Touyama Gaming").players))
ZONE_GRID = np.asarray(parse_layer(MACRO_ZONE_STR))
A_SITE_CELLS = frozenset(tuple(map(int, p)) for p in
                         zip(*np.where(ZONE_GRID == ZONE_MARKERS["A_SITE"])))


def forced_attack_site(game):
    owner = getattr(game, "real_game", game)
    roster = getattr(owner, "defender_roster", None)
    if roster is None:
        # Include dead players: casualties must not change team identity.
        roster = [c.name for c in getattr(owner, "chars", []) if c.team == "D"]
    names = tuple(map(str, roster))
    return "A" if len(names) == len(TOUYAMA_ROSTER) and frozenset(names) == TOUYAMA_ROSTER else None


def attack_plant_cells(game, grid):
    cells = [tuple(map(int, p)) for p in zip(*np.where(np.asarray(grid) == 2))]
    if forced_attack_site(game) == "A":
        cells = [p for p in cells if p in A_SITE_CELLS]
    return cells


def enforce_attack_target(game, game_state=None):
    """Keep public shared intent on A, even when an old target/commit is B."""
    if forced_attack_site(game) is None:
        return
    if bool((game_state or {}).get("is_planted", getattr(game, "is_planted", False))):
        return
    grid = (game_state or {}).get("grid", getattr(game, "grid", None))
    if grid is None:
        return
    cells = attack_plant_cells(game, grid)
    if not cells:
        return
    target = team_plant_target(game)
    if target not in cells:
        chars = (game_state or {}).get("chars", getattr(game, "chars", []))
        holder = next((c for c in chars if c.team == "A" and c.is_alive
                       and getattr(c, "has_spike", False)), None)
        origin = holder.pos if holder is not None else cells[0]
        target = min(cells, key=lambda p: (abs(p[0] - origin[0]) + abs(p[1] - origin[1]), p))
        set_team_plant_target(game, target)
    if game_state is not None:
        game_state["target_plant_pos"] = target
