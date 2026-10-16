"""Fire at recon-disclosed enemies; take cover when our position is revealed."""

from collections import deque

from ov1_inference_carry_escape import escape_disclosed_fire
from ov1_inference_damage_avoidance import danger_cells
from ov1_inference_team_tactics import facing_towards, known_enemies, shot_clear, tactical_state
from game_core import DEFUSE_REQUIRED_TICKS
from grid_lines import line_cells

CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DEFUSE_PRIORITY_MARGIN_TICKS = 4


def _defuse_is_urgent(char, game_state):
    if (char.team != "D" or not game_state.get("is_planted")
            or game_state.get("planted_pos") is None
            or "detonate_timer" not in game_state):
        return False

    if getattr(char, "defuse_timer", 0) > 0:
        return True

    defuse_info = game_state.get("defender_defuse_info") or {}
    active_defuser = next(
        (name for name, (timer, _) in defuse_info.items() if timer > 0), None
    )
    if active_defuser == char.name:
        return True
    if active_defuser is not None:
        return False

    grid = game_state["grid"]
    start = tuple(map(int, char.pos))
    spike = tuple(map(int, game_state["planted_pos"]))
    goals = {
        (row, col)
        for row in range(max(0, spike[0] - 1), min(grid.shape[0], spike[0] + 2))
        for col in range(max(0, spike[1] - 1), min(grid.shape[1], spike[1] + 2))
        if grid[row, col] != 1
    }
    if not goals:
        return False

    queue = deque([(start, 0)])
    visited = {start}
    distance = None
    while queue:
        cell, steps = queue.popleft()
        if cell in goals:
            distance = steps
            break
        for dr, dc in CARDINAL:
            nxt = (cell[0] + dr, cell[1] + dc)
            if (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                    and grid[nxt] != 1 and nxt not in visited):
                visited.add(nxt)
                queue.append((nxt, steps + 1))

    if distance is None:
        return False
    return float(game_state["detonate_timer"]) <= (
        distance + DEFUSE_REQUIRED_TICKS + DEFUSE_PRIORITY_MARGIN_TICKS
    )


def _flash_push_active(char, state):
    return (
        char.team == "D"
        and state.get("is_planted")
        and any(
            getattr(enemy, "is_alive", True)
            and enemy.team != char.team
            and getattr(enemy, "blind_remaining", 0) > 0
            for enemy in state.get("chars", ())
        )
    )


def recon_response(char, game_state, memory):
    if not getattr(char, "is_alive", True) or game_state.get("defender_setup_active"):
        return None
    state = tactical_state(game_state)
    enemies = known_enemies(char, state)
    if getattr(char, "reveal_remaining", 0) > 0:
        if _defuse_is_urgent(char, game_state):
            if (getattr(char, "defuse_timer", 0) > 0
                    or max(abs(int(game_state["planted_pos"][0]) - int(char.pos[0])),
                           abs(int(game_state["planted_pos"][1]) - int(char.pos[1]))) <= 1):
                return list(char.pos), "DEFUSE"
            memory.clear()
            return None
        if _flash_push_active(char, state):
            memory.clear()
            return None
        escape = escape_disclosed_fire(char, game_state, memory)
        if escape is not None:
            return escape
        if enemies:
            # Known attackers cannot shoot through the current wall/body cover.
            enemy = min(enemies, key=lambda e: abs(e.pos[0]-char.pos[0])+abs(e.pos[1]-char.pos[1]))
            return list(char.pos), {"facing": facing_towards(char.pos, enemy.pos) or char.facing}
        # A recon hit does not reveal the shooter's position to its victim.
        # Without a disclosed attacker, seek nearby wall cover using only map
        # geometry. Do not read hidden enemy coordinates to choose a direction.
        grid = state["grid"]
        occupied = {tuple(c.pos) for c in state["chars"] if c.name != char.name and c.is_alive
                    and (c.team == char.team or getattr(c, "position_known", True))}
        danger = danger_cells(char, game_state)

        def wall(cell):
            return not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]) or grid[cell] == 1

        start = tuple(char.pos)
        candidates = [(start[0]+dr, start[1]+dc) for dr, dc in CARDINAL]
        candidates = [c for c in candidates if not wall(c) and c not in occupied and c not in danger]
        previous = memory.get("unknown_previous")
        def rank(cell):
            return (-sum(wall((cell[0]+dr,cell[1]+dc)) for dr,dc in CARDINAL), cell == previous, cell)
        step = min(candidates, key=rank, default=start)
        memory["unknown_previous"] = start
        char._navigation_step_limit = 1
        return list(step), {"facing": getattr(char, "facing", "N")}
    memory.clear()
    # Preserve planting and postplant positioning/defusing. These actors still
    # face disclosed enemies through shared tactics and the engine auto-fires
    # whenever their accepted action permits it. Do not stall the objective
    # throughout a distant recon window just to request stationary fire.
    if getattr(char, "has_spike", False) or game_state.get("is_planted"):
        return None
    smoke = set(state.get("smoke_cells") or ())
    targets = [e for e in enemies if getattr(e, "reveal_remaining", 0) > 0
               and len(line_cells(tuple(char.pos), tuple(e.pos))) > 2
               and smoke.intersection(line_cells(tuple(char.pos), tuple(e.pos)))
               and shot_clear(char, e, state)]
    if targets:
        enemy = min(targets, key=lambda e: abs(e.pos[0]-char.pos[0])+abs(e.pos[1]-char.pos[1]))
        return list(char.pos), {"facing": facing_towards(char.pos, enemy.pos) or char.facing}
    return None
