"""Face disclosed team sightings and interrupt defuses using engine LOS rules."""
import math
from collections import deque
from types import SimpleNamespace
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


def shot_clear(char, enemy, state, position=None, *, block_players=True):
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
    if not block_players:
        return True
    intermediate = set(cells[1:-1])
    return not any(other.is_alive and other.name not in (char.name, enemy.name)
                   and tuple(other.pos) in intermediate for other in state.get("chars", ()))


def search_hidden_defuser(char, state, memory):
    """Check legal tap cells using public geometry, never hidden enemy positions.

    A tapper cannot move while maintaining defuse progress. Remember cells
    checked during this tap so reaching the spike does not end the search.
    """
    state = {**state, **tactical_state(state)}
    taps = state.get("defender_defuse_info") or {}
    active = next(((name, timer) for name, (timer, _) in taps.items() if timer > 0), None)
    if not state.get("is_planted") or state.get("planted_pos") is None or active is None:
        memory.clear()
        return None
    name, timer = active
    if any(enemy.name == name for enemy in known_enemies(char, state)):
        memory.clear()
        return None
    grid = state["grid"]
    spike, start = tuple(state["planted_pos"]), tuple(char.pos)
    key = (state.get("current_round"), spike, name)
    candidates = {(r, c) for r in range(spike[0] - 1, spike[0] + 2)
                  for c in range(spike[1] - 1, spike[1] + 2)
                  if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1}
    if memory.get("key") != key or timer < memory.get("timer", 0):
        memory.clear()
        memory.update(key=key, remaining=set(candidates))
    memory["timer"] = timer
    # Only disclosed bodies may affect route selection and hypothetical shots.
    disclosed = [other for other in state.get("chars", ()) if other.is_alive
                 and (other.team == char.team or getattr(other, "position_known", True))]
    public = {**state, "chars": disclosed}
    occupied = {tuple(other.pos) for other in disclosed if other.name != char.name}

    def target(cell):
        return SimpleNamespace(name="__defuse_search_cell__", pos=cell, reveal_remaining=0)

    remaining = memory["remaining"]
    if not getattr(char, "blind_remaining", 0):
        fx, fy = FACING_VECTORS[getattr(char, "facing", "N")]
        remaining.difference_update(cell for cell in tuple(remaining)
                                    if fx * (cell[1] - start[1]) + fy * (cell[0] - start[0]) >= 0
                                    and shot_clear(char, target(cell), public, block_players=False))
    if not remaining:
        # A new scan is preferable to stopping permanently after one sweep.
        remaining.update(candidates)
    # Prioritize going around actual blind corners over turning toward an
    # already exposed cell behind us at the spike.
    occluded = {point for point in remaining
                if not shot_clear(char, target(point), public, block_players=False)}
    search_points = occluded or remaining
    queue, steps, distances = deque([start]), {start: start}, {start: 0}
    goals, goal_distance = [], None
    while queue:
        cell = queue.popleft()
        if goal_distance is not None and distances[cell] > goal_distance:
            break
        shootable = [point for point in search_points
                     if point != cell and shot_clear(char, target(point), public, cell)]
        if shootable:
            goal_distance = distances[cell]
            goals.append((cell, shootable))
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                    and grid[nxt] != 1 and nxt not in occupied and nxt not in steps):
                steps[nxt] = nxt if cell == start else steps[cell]
                distances[nxt] = distances[cell] + 1
                queue.append(nxt)
    if goals:
        cell, shootable = min(goals, key=lambda goal: (-len(goal[1]), math.dist(goal[0], spike), goal[0]))
        point = min(shootable, key=lambda p: (math.dist(cell, p), p))
        step = steps[cell]
        facing = facing_towards(step, point) or getattr(char, "facing", "N")
        # A speed bonus must not carry us past the peek or a corner.
        char._navigation_step_limit = 1
        if not getattr(char, "facing_forced_this_tick", False):
            char.facing = facing
        return list(step), {"facing": facing}
    return None


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
