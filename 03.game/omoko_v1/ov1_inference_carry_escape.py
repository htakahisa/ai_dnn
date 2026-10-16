"""Omoko's awakened carrier uses disclosed threats to find nearby cover."""

from collections import deque
from types import SimpleNamespace

from ov1_inference_damage_avoidance import danger_cells
from ov1_inference_team_tactics import facing_towards, known_enemies, shot_clear, tactical_state

ESCAPE_AWAKENING_NAME = "にげるっすー!"
ESCAPE_SEARCH_DISTANCE = 8
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def awakened_carry_escape(char, game_state, memory=None):
    memory = {} if memory is None else memory
    key = (game_state.get("current_round"), char.name)
    if (not getattr(char, "is_alive", True) or not getattr(char, "has_spike", False)
            or game_state.get("is_planted") or game_state.get("defender_setup_active")
            or str(getattr(char, "base_name", char.name)) != "おもこ"
            or ESCAPE_AWAKENING_NAME not in getattr(char, "active_awakenings", {})):
        memory.pop(key, None)
        return None
    return escape_disclosed_fire(char, game_state, memory)


def escape_disclosed_fire(char, game_state, memory=None):
    """Shared escape planner; caller decides whether a retreat is warranted."""
    memory = {} if memory is None else memory
    key = (game_state.get("current_round"), char.name)
    state = tactical_state(game_state)
    enemies = known_enemies(char, state)
    if not enemies:
        if key in memory:
            return list(char.pos), {"facing": getattr(char, "facing", "N")}
        return None
    grid = state["grid"]
    start = tuple(char.pos)
    # Use only the carrier's public state at a hypothetical destination.
    target = SimpleNamespace(name=char.name, team=char.team, is_alive=True,
                             pos=start, reveal_remaining=getattr(char, "reveal_remaining", 0))
    exposure_cache = {}

    def threats(cell):
        if cell not in exposure_cache:
            target.pos = cell
            exposure_cache[cell] = tuple(e for e in enemies if shot_clear(e, target, state))
        return exposure_cache[cell]

    current_threats = threats(start)
    if not current_threats:
        if key in memory:
            enemy = min(enemies, key=lambda e: abs(e.pos[0] - start[0]) + abs(e.pos[1] - start[1]))
            return list(start), {"facing": facing_towards(start, enemy.pos) or char.facing}
        return None
    memory[key] = True
    allies = [c for c in state.get("chars", ()) if c.team == char.team
              and c.name != char.name and c.is_alive]
    occupied = {tuple(c.pos) for c in state.get("chars", ())
                if c.name != char.name and c.is_alive
                and (c.team == char.team or getattr(c, "position_known", True))}
    danger = danger_cells(char, game_state)

    def gap(cell):
        return min(abs(cell[0] - e.pos[0]) + abs(cell[1] - e.pos[1]) for e in current_threats)

    def support(cell):
        return min((abs(cell[0] - a.pos[0]) + abs(cell[1] - a.pos[1]) for a in allies), default=0)

    # Search shortest paths to cover. Within one distance layer prefer less
    # exposure and less approach toward the attackers, then nearby allies.
    queue = deque([start])
    paths = {start: ()}
    costs = {start: (0, 0)}
    covers = []
    cover_distance = None
    while queue:
        cell = queue.popleft()
        path = paths[cell]
        if cover_distance is not None and len(path) > cover_distance:
            break
        if path and not threats(cell):
            covers.append(cell)
            cover_distance = len(path)
            continue
        if len(path) >= ESCAPE_SEARCH_DISTANCE or cover_distance is not None:
            continue
        for dr, dc in CARDINAL:
            nxt = cell[0] + dr, cell[1] + dc
            if (not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1])
                    or grid[nxt] == 1 or nxt in occupied or nxt in danger):
                continue
            cost = (costs[cell][0] + len(threats(nxt)),
                    costs[cell][1] + max(0, gap(cell) - gap(nxt)))
            if nxt in paths:
                if len(paths[nxt]) != len(path) + 1 or cost >= costs[nxt]:
                    continue
                # Same-layer parents are processed before this child is popped.
                paths[nxt], costs[nxt] = path + (nxt,), cost
                continue
            paths[nxt], costs[nxt] = path + (nxt,), cost
            queue.append(nxt)

    if covers:
        goal = min(covers, key=lambda cell: (costs[cell], support(cell), -gap(cell), cell))
        path = paths[goal]
    else:
        # No reachable cover: reduce exposure or gain distance rather than
        # falling back to a forward rush toward the plant goal.
        options = [cell for cell, path in paths.items() if len(path) == 1
                   and (len(threats(cell)), -gap(cell)) < (len(current_threats), -gap(start))]
        goal = min(options, key=lambda cell: (len(threats(cell)), -gap(cell), support(cell), cell),
                   default=start)
        path = paths[goal]

    step = path[0] if path else start
    direction = step[0] - start[0], step[1] - start[1]
    straight = 0
    previous = start
    for cell in path:
        if (cell[0] - previous[0], cell[1] - previous[1]) != direction:
            break
        straight += 1
        previous = cell
    # Existing BaseController navigation uses this engine-owned per-move cap.
    # It prevents the awakened extra steps from overshooting cover or a turn.
    char._navigation_step_limit = max(1, straight)
    enemy = min(current_threats, key=lambda e: abs(e.pos[0] - step[0]) + abs(e.pos[1] - step[1]))
    facing = facing_towards(step, enemy.pos) or getattr(char, "facing", "N")
    return list(step), {"facing": facing}
