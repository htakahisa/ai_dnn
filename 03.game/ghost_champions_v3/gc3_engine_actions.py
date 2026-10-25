"""Game command encoding and physical permissions only; no tactical preferences."""
import math
import numpy as np
from game_core import RAMP_ELECTRIC_TICKS, ASH_RANGE_CELLS, DANCE_MAX_HP
from map_data_defender_setup import get_setup_mask

MODES = ("MOVE", "WATCH", "MOVE_WATCH", "ABILITY", "ULTIMATE", "PLANT", "DEFUSE", "COLLECT_ORB")
FACING_DIRS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
ABILITIES = ("RECON", "FLASH", "SMOKE", "RAMP", "DANCE", "ASH", "HUNT", "UNKNOWN")
ULTIMATES = ("MONITOR", "TUNNEL", "ESCAPE", "NEON", "SERENADE", "BALEMOON", "RAID", "UNKNOWN")


def valid(grid, point):
    r, c = point
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1


def facing_towards(source, target, fallback="S"):
    dr, dc = target[0]-source[0], target[1]-source[1]
    if dr == dc == 0:
        return fallback
    # Quantize the sampled target to the engine's eight direction commands.
    index = int(math.floor((math.atan2(dc, -dr)+math.pi/8)/(math.pi/4))) % 8
    return FACING_DIRS[index]


def capabilities(char, state):
    alive = bool(char.is_alive)
    channeling = any(p.get("owner") == char.name for p in state.get("ally_escape_portals", ()))
    channeling |= char.name in state.get("visible_escape_channel_owners", ())
    can_act = alive and not channeling and not getattr(char, "movement_disabled_remaining", 0)
    applied = getattr(char, "electric_applied_tick", None)
    electric = getattr(char, "electric_remaining", 0) > 0 and (
        applied is None or state.get("battle_tick", 0)+1 < applied+RAMP_ELECTRIC_TICKS)
    can_move = can_act and not electric and not getattr(char, "fate_loom_remaining", 0)
    can_cast = can_act and not getattr(char, "ability_seal_remaining", 0)
    return dict(can_act=bool(can_act), can_move=bool(can_move), can_cast=bool(can_cast),
                can_turn=bool(can_act and not getattr(char, "facing_forced_this_tick", False)))


def allies(char, state):
    return sorted((c for c in state.get("chars", ()) if c.team == char.team), key=lambda c: str(c.name))[:5]


def projectile_targets(grid, origin):
    """Match the first Bresenham step of the engine's extended projectile ray.

    A wall later on the ray is a legal impact; only an immediately blocked
    ray (or aiming at oneself) makes FLASH/RECON fail to launch.
    """
    coords = np.indices(grid.shape).reshape(2, -1).T
    delta = coords-np.asarray(origin)
    distance = np.abs(delta)
    step = np.sign(delta)
    step[:, 0] *= 2*distance[:, 0] >= distance[:, 1]
    step[:, 1] *= 2*distance[:, 1] >= distance[:, 0]
    first = np.asarray(origin)+step
    inside = (first >= 0).all(axis=1) & (first < np.asarray(grid.shape)).all(axis=1)
    allowed = np.zeros(grid.size, dtype=np.bool_)
    indices = np.flatnonzero(inside & (distance.sum(axis=1) > 0))
    allowed[indices] = grid[first[indices, 0], first[indices, 1]] != 1
    return allowed & (grid != 1).reshape(-1)


def legal_actions(char, state):
    grid = np.asarray(state["grid"])
    cells = grid.size
    origin = tuple(map(int, char.pos))
    caps = capabilities(char, state)
    setup = bool(state.get("defender_setup_active"))
    occupied = {tuple(c.pos) for c in state.get("chars", ()) if c.name != char.name and c.is_alive
                and (c.team == char.team or (not setup and getattr(c, "position_known", False)))}
    move = np.zeros(grid.shape, dtype=np.bool_)
    move[origin] = True
    if caps["can_move"]:
        setup_mask = np.asarray(get_setup_mask()) if setup else None
        for dr, dc in CARDINAL:
            point = origin[0]+dr, origin[1]+dc
            if valid(grid, point) and point not in occupied and (setup_mask is None or setup_mask[point] == 0):
                move[point] = True
    floor = (grid != 1).reshape(-1)
    watch = floor.copy()  # Watching through a wall is an intention, not an illegal cast.
    target = floor.copy()
    ultimate_target = floor.copy()
    ability = getattr(char, "ability_name", "UNKNOWN")
    if ability in ("FLASH", "RECON"):
        target = projectile_targets(grid, origin)
    if ability == "ASH":
        coords = np.indices(grid.shape).reshape(2, -1).T
        target &= np.sum((coords-np.array(origin))**2, axis=-1) <= ASH_RANGE_CELLS**2
    if getattr(char, "ultimate_name", None) == "ESCAPE":
        for point in occupied:
            if valid(grid, point):
                ultimate_target[np.ravel_multi_index(point, grid.shape)] = False
    healing = np.zeros(5, dtype=np.bool_)
    for i, ally in enumerate(allies(char, state)):
        reduced = getattr(ally, "contract_max_hp_lost", 0) or getattr(ally, "fate_max_hp_lost", 0)
        cap = min(DANCE_MAX_HP, getattr(ally, "max_hp", 100)) if reduced else DANCE_MAX_HP
        healing[i] = ally.name != char.name and ally.is_alive and ally.hp < cap
    modes = np.zeros(len(MODES), dtype=np.bool_)
    modes[0] = True
    if caps["can_act"]:
        modes[1] = caps["can_turn"]
        modes[2] = caps["can_move"] and caps["can_turn"] and move.sum() > 1
        if not setup:
            available = ability in ("RECON", "FLASH", "SMOKE", "RAMP", "DANCE", "ASH")
            modes[3] = caps["can_cast"] and available and getattr(char, ability.lower()+"_charges", 0) > 0
            modes[3] &= bool(healing.any()) if ability == "DANCE" else bool(target.any()) if ability != "RAMP" else True
            ultimate = getattr(char, "ultimate_name", "UNKNOWN")
            modes[4] = caps["can_cast"] and ultimate in ULTIMATES[:-1] and ultimate != "SERENADE" and (
                getattr(char, "ultimate_points", 0) >= getattr(char, "ultimate_cost", 1))
            if ultimate in ("RAID", "ESCAPE"):
                modes[4] &= caps["can_move"]
            if ultimate in ("NEON", "ESCAPE"):
                modes[4] &= bool(ultimate_target.any())
            modes[5] = char.team == "A" and getattr(char, "has_spike", False) and not state.get("is_planted") and grid[origin] == 2
            plant = state.get("planted_pos")
            modes[6] = char.team == "D" and bool(state.get("is_planted")) and plant is not None and max(
                abs(origin[i]-plant[i]) for i in (0, 1)) <= 1
            modes[7] = origin in state.get("available_orbs", ())
    if not caps["can_act"]:
        move[:] = False
        move[origin] = True
    return dict(mode=modes, position=move.reshape(cells), watch=watch, target=target,
                ultimate_target=ultimate_target, ally=healing), caps


def decode_command(char, state, selections):
    mode = MODES[selections["mode"]]
    point = lambda key: tuple(map(int, np.unravel_index(selections[key], state["grid"].shape)))
    if mode in ("MOVE", "MOVE_WATCH"):
        destination = list(point("position"))
        if mode == "MOVE_WATCH":
            return destination, {"facing": facing_towards(destination, point("watch"), char.facing)}
        return destination, "MOVE"
    if mode == "WATCH":
        return list(char.pos), {"facing": facing_towards(char.pos, point("watch"), char.facing)}
    if mode == "ABILITY":
        ability = char.ability_name
        command = {"ability": ability}
        if ability == "DANCE":
            command["target_name"] = allies(char, state)[selections["ally"]].name
        elif ability != "RAMP":
            command["target"] = point("target")
        return list(char.pos), command
    if mode == "ULTIMATE":
        command = {"ultimate": char.ultimate_name}
        if char.ultimate_name in ("NEON", "ESCAPE"):
            command["target"] = point("target")
        elif char.ultimate_name in ("RAID", "TUNNEL"):
            command["facing"] = facing_towards(char.pos, point("watch"), char.facing)
        return list(char.pos), command
    return list(char.pos), mode
