"""Shared inference safety for public ASH/BALEMOON destruction areas.

No enemy positions or training models are consulted. The engine still owns
damage, contract duration, collision, and forced facing.
"""
from collections import deque

AVOID_DESTRUCTION_AREAS = True
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def danger_cells(char, game_state, game=None):
    if not AVOID_DESTRUCTION_AREAS:
        return set()
    effects = game_state.get("public_effects")
    if effects is None:
        if game is None:
            return set()
        from public_effects import PublicEffectReader
        effects = PublicEffectReader().read_visible(game, char.team)
    # The public sensor exports visible cells, not owners or hidden timers.
    # Treat all displayed destruction cells and BALEMOON warnings cautiously.
    return {tuple(cell) for effect in effects
            if effect.kind == "DESTRUCTION" or (effect.kind == "BALEMOON" and effect.phase == "warning")
            for cell in effect.cells}


def _first_step(char, game_state, danger, goal=None):
    """Escape by shortest walk, or route to a goal without entering danger."""
    grid = game_state["grid"]
    start = tuple(char.pos)
    escaping = start in danger
    occupied = {tuple(other.pos) for other in game_state.get("chars", ())
                if other.name != char.name and other.is_alive}
    queue = deque([start])
    steps = {start: start}
    while queue:
        cell = queue.popleft()
        if (escaping and cell not in danger) or (not escaping and cell == goal):
            return steps[cell]
        for dr, dc in CARDINAL:
            nxt = (cell[0] + dr, cell[1] + dc)
            if (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                    and grid[nxt] != 1 and nxt not in occupied and nxt not in steps
                    and (escaping or nxt not in danger)):
                steps[nxt] = nxt if cell == start else steps[cell]
                queue.append(nxt)
    return start


def _move_request(char, step):
    # Cancel planting/defusing/utility rather than retaining a stationary action.
    return list(step), {"facing": getattr(char, "facing", "N")}


def escape_destruction_area(char, game_state, game=None):
    danger = danger_cells(char, game_state, game)
    if not getattr(char, "is_alive", True) or tuple(char.pos) not in danger:
        return None
    return _move_request(char, _first_step(char, game_state, danger))


def avoid_destruction_entry(char, game_state, result, game=None):
    danger = danger_cells(char, game_state, game)
    if not danger or not getattr(char, "is_alive", True):
        return result
    # Controllers return a position or (position, action payload).
    position = result[0] if isinstance(result, tuple) else result
    proposed = tuple(map(int, position))
    if proposed not in danger:
        return result
    start = tuple(char.pos)
    dr, dc = proposed[0] - start[0], proposed[1] - start[1]
    if (dr, dc) not in CARDINAL:
        return _move_request(char, start)
    # Navigate around the zone toward its far edge in the intended direction.
    # Do not make danger globally walkable to preserve an otherwise short path.
    goal = proposed
    grid = game_state["grid"]
    while goal in danger:
        goal = (goal[0] + dr, goal[1] + dc)
        if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]) or grid[goal] == 1:
            return _move_request(char, start)
    return _move_request(char, _first_step(char, game_state, danger, goal))
