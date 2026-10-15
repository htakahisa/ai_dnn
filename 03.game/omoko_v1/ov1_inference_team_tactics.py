"""Face disclosed team sightings and interrupt defuses using engine LOS rules."""
import math
from grid_lines import line_cells, wall_line_cells
from game_core import FACING_VECTORS

USE_SHARED_ENEMY_FACING = True


def tactical_state(state):
    return state.get("team_tactical_state", state)


def known_enemies(char, state):
    grid = state["grid"]
    return [enemy for enemy in state.get("chars", ())
            if enemy.is_alive and enemy.team != char.team
            and getattr(enemy, "position_known", True)
            and 0 <= enemy.pos[0] < grid.shape[0] and 0 <= enemy.pos[1] < grid.shape[1]]


def facing_towards(start, end):
    dc, dr = end[1] - start[1], end[0] - start[0]
    if dc == 0 and dr == 0:
        return None
    return max(FACING_VECTORS, key=lambda direction:
               FACING_VECTORS[direction][0] * dc + FACING_VECTORS[direction][1] * dr)


def shot_clear(char, enemy, state, position=None):
    """Match wall supercover, adjacent smoke visibility and body blocking."""
    start = tuple(char.pos) if position is None else tuple(position)
    end = tuple(enemy.pos)
    grid = state["grid"]
    for r, c in wall_line_cells(start, end):
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or grid[r, c] == 1:
            return False
    cells = line_cells(start, end)
    ignores_smoke = (getattr(char, "sees_through_smoke", False)
                     or getattr(enemy, "reveal_remaining", 0) > 0)
    if not ignores_smoke and len(cells) > 2 and set(cells).intersection(state.get("smoke_cells") or ()):
        return False
    intermediate = set(cells[1:-1])
    return not any(other.is_alive and other.name not in (char.name, enemy.name)
                   and tuple(other.pos) in intermediate for other in state.get("chars", ()))


def apply_shared_facing(char, state, result):
    if not USE_SHARED_ENEMY_FACING or not char.is_alive or getattr(char, "facing_forced_this_tick", False):
        return result
    state = tactical_state(state)
    enemies = known_enemies(char, state)
    if not enemies:
        return result
    position = result[0] if isinstance(result, tuple) else result
    payload = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    # Preserve the model's directional utility/ultimate decision.
    if isinstance(payload, dict) and ("ability" in payload or "ultimate" in payload):
        return result
    taps = state.get("defender_defuse_info") or {}
    if any(timer > 0 for timer, _ in taps.values()) and not any(
            taps.get(enemy.name, (0, 0))[0] > 0 for enemy in enemies):
        # Preserve the guard's search rotation when the tapper is still hidden.
        return result
    def priority(enemy):
        tap = taps.get(enemy.name, (0, 0))[0] > 0
        return (not tap, not shot_clear(char, enemy, state, position),
                math.hypot(enemy.pos[0] - position[0], enemy.pos[1] - position[1]), enemy.name)
    target = min(enemies, key=priority)
    facing = facing_towards(position, target.pos) or getattr(char, "facing", "N")
    char.facing = facing
    if isinstance(payload, str):
        # PLANT/DEFUSE/COLLECT_ORB require their string action to be preserved.
        return result
    return list(position), {**(payload or {}), "facing": facing}
