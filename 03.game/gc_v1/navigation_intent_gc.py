"""Own-team Macro intent features shared by actual-game learning and inference.

Macro chooses roles and waypoints. These helpers expose that plan to a policy;
they never select movement actions or read hidden opponent positions.
"""
from collections import deque

import numpy as np

from game_core import SHOOTING_SITE_DIGREE
try:
    from .positioning_gc import team_plant_target
except ImportError:
    from positioning_gc import team_plant_target

INTENT_DIM = 8
NAVIGATION_CONTEXT_DIM = 18
SCREENING_DIM = 4
FORMATION_DIM = 4
ROUTE_CLEARANCE_DIM = 4
SAFE_SCREEN_ADVANCE_DIM = 4
FORMATION_LEAD_STEPS = 3
FORMATION_READY_RADIUS = 1
FORMATION_MIN_FINAL_DISTANCE = 2
FORMATION_MAX_FINAL_DISTANCE = 24
ENTRY_SYNC_STAGING_DISTANCE = 8
ENTRY_SYNC_MAX_FORMATION_DISTANCE = 6
ENTRY_SYNC_MIN_ROUND_TIME = 15
ENTRY_SYNC_MAX_IDLE_FEATURE = 0.10
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
INDEPENDENT_ROLES = {"SUPPORT", "FAKE_SELL", "OPPOSITE_SCOUT", "OPPOSITE_SCOUT_LURK",
                     "OPPOSITE_SCOUT_DEEP", "MID_CONTROL", "A_SCOUT", "B_SCOUT"}
SCREENING_ROLES = {"MAIN", "MAIN_LEAD", "DEFAULT_MAIN", "ROTATE", "REHIT"}


def _distance_map(game, goal, cache):
    if goal not in cache:
        distance = np.full(game.grid.shape, -1, dtype=np.int32)
        distance[goal] = 0
        queue = deque([goal])
        while queue:
            pos = queue.popleft()
            for dr, dc in CARDINAL:
                nxt = (pos[0] + dr, pos[1] + dc)
                if (0 <= nxt[0] < distance.shape[0] and 0 <= nxt[1] < distance.shape[1]
                        and game.grid[nxt] != 1 and distance[nxt] < 0):
                    distance[nxt] = distance[pos] + 1
                    queue.append(nxt)
        cache[goal] = distance
    return cache[goal]


def own_macro(game):
    real = getattr(game, "real_game", game)
    controller = getattr(real, "attacker_controller", None)
    seen = set()
    while controller is not None and id(controller) not in seen:
        seen.add(id(controller))
        macro = getattr(controller, "macro_controller", None)
        if macro is not None:
            return macro
        controller = getattr(controller, "inner_controller", None)
    return None


def navigation_intent(game, char):
    macro = own_macro(game)
    env = getattr(macro, "env", None)
    strategy = str(getattr(env, "current_strategy", "") or "")
    assignment = getattr(env, "assignment", {}).get(char.name, (None, None, "MAIN"))
    role = assignment[2] if len(assignment) > 2 else "MAIN"
    goal = getattr(env, "targets", {}).get(char.name)
    plant = team_plant_target(game)
    if goal is None:
        goal = plant
    if goal is not None:
        goal = tuple(map(int, goal))
        grid = game.grid
        if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]) or grid[goal] == 1:
            goal = plant
        # Macro's final plant waypoint can be an ordinary floor. Carry's final
        # task is the registered own-team plant target on the same site.
        elif getattr(char, "has_spike", False) and plant is not None and grid[goal] == 2:
            if (goal[1] < grid.shape[1] // 2) == (plant[1] < grid.shape[1] // 2):
                goal = plant
    return (tuple(goal) if goal is not None else None), strategy, role


def can_engage(game, char, chars):
    """Whether automatic shooting has a current target, using the given view."""
    if not getattr(char, "is_alive", True) or getattr(char, "plant_timer", 0) > 0:
        return False
    for enemy in chars:
        if enemy.team == char.team or not getattr(enemy, "is_alive", True):
            continue
        if not game.check_line_of_sight(char, enemy):
            continue
        intermediate = set(game._line_cells(tuple(char.pos), tuple(enemy.pos))[1:-1])
        if any(other.name not in (char.name, enemy.name) and getattr(other, "is_alive", True)
               and tuple(other.pos) in intermediate for other in chars):
            continue
        if game._facing_angle_diff(char, enemy) <= SHOOTING_SITE_DIGREE:
            return True
    return False


def intent_features(game, char, chars, cache):
    goal, strategy, role = navigation_intent(game, char)
    features = np.zeros(INTENT_DIM, dtype=np.float32)
    if goal is not None:
        if goal not in cache:
            grid = game.grid
            distance = np.full(grid.shape, -1, dtype=np.int32)
            distance[goal] = 0
            queue = deque([goal])
            while queue:
                pos = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nxt = (pos[0] + dr, pos[1] + dc)
                    if (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                            and grid[nxt] != 1 and distance[nxt] < 0):
                        distance[nxt] = distance[pos] + 1
                        queue.append(nxt)
            cache[goal] = distance
        distance = cache[goal]
        pos = tuple(map(int, char.pos))
        current = int(distance[pos])
        features[0] = max(0, current) / sum(game.grid.shape)
        best = current
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (pos[0] + dr, pos[1] + dc)
            if (0 <= nxt[0] < distance.shape[0] and 0 <= nxt[1] < distance.shape[1]
                    and 0 <= distance[nxt] < best):
                best = distance[nxt]
                features[1:3] = (dr, dc)
    features[3:6] = ("SPLIT" in strategy, "FAKE" in strategy, "ROTATE" in strategy)
    features[6] = role in INDEPENDENT_ROLES
    features[7] = can_engage(game, char, chars)
    return features


def navigation_context_features(game, char, chars, distance, history, goal):
    """Observable neighbor costs and executed movement history, never actions.

    Four distance gains, four walls, four occupied cells, the last actual
    displacement, time without new progress, revisits, deadline slack and
    waypoint arrival. Histories belong to individual actors and round resets.
    """
    features = np.zeros(NAVIGATION_CONTEXT_DIM, dtype=np.float32)
    grid = game.grid
    pos = tuple(map(int, char.pos))
    current = int(distance[pos]) if distance is not None else -1
    occupied = {tuple(map(int, c.pos)) for c in chars
                if c.name != char.name and getattr(c, "is_alive", True)}
    for i, (dr, dc) in enumerate(CARDINAL):
        nxt = (pos[0] + dr, pos[1] + dc)
        wall = not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]) or grid[nxt] == 1
        features[4 + i] = wall
        features[8 + i] = nxt in occupied
        reachable = not wall and distance is not None and int(distance[nxt]) >= 0 and current >= 0
        features[i] = np.clip(current - int(distance[nxt]), -1, 1) if reachable else -1

    tick = int(getattr(game, "battle_tick", 0))
    state = history.get(char.name)
    if state is None or state["goal"] != goal or tick < state["tick"]:
        state = {"goal": goal, "pos": pos, "tick": tick, "best": current,
                 "progress_tick": tick, "visits": deque(maxlen=8), "delta": (0, 0)}
        history[char.name] = state
    elif tick != state["tick"]:
        state["delta"] = (pos[0] - state["pos"][0], pos[1] - state["pos"][1])
        if current >= 0 and (state["best"] < 0 or current < state["best"]):
            state["best"] = current
            state["progress_tick"] = tick
        state["visits"].append(state["pos"])
        state["pos"], state["tick"] = pos, tick
    features[12:14] = np.clip(state["delta"], -1, 1)
    features[14] = min(1.0, (tick - state["progress_tick"]) / 20.0)
    features[15] = sum(p == pos for p in state["visits"]) / 8.0
    from game_core import PLANT_REQUIRED_TICKS
    remaining = int(getattr(game, "round_timer", 100))
    features[16] = np.clip((remaining - max(0, current) - PLANT_REQUIRED_TICKS) / sum(grid.shape), -1, 1)
    features[17] = 0 <= current <= 1
    return features


def carrier_screening_status(game, carrier, chars, cache):
    """Observable own-team screen around the carrier's current Macro route."""
    route_goal = navigation_intent(game, carrier)[0]
    final_goal = team_plant_target(game)
    if route_goal is None:
        route_goal = final_goal
    if route_goal is None:
        return {"route_goal": None, "final_distance": -1, "ahead": [], "near": []}
    route_distance = _distance_map(game, route_goal, cache)
    carrier_pos = tuple(map(int, carrier.pos))
    current = int(route_distance[carrier_pos])
    # A synchronization waypoint is considered reached within one cell. Use
    # the final plant route for screening direction while the carrier waits,
    # otherwise no teammate could have a distance smaller than zero/one.
    if current <= 1 and final_goal is not None and final_goal != route_goal:
        route_goal = final_goal
        route_distance = _distance_map(game, route_goal, cache)
        current = int(route_distance[carrier_pos])
    allies = [c for c in chars if c.name != carrier.name and c.team == carrier.team
              and getattr(c, "is_alive", True)]
    near = [c for c in allies if max(abs(c.pos[0] - carrier_pos[0]),
                                     abs(c.pos[1] - carrier_pos[1])) <= 4]
    ahead = [c for c in allies if current >= 0
             and 0 <= int(route_distance[tuple(map(int, c.pos))]) < current
             and max(abs(c.pos[0] - carrier_pos[0]), abs(c.pos[1] - carrier_pos[1])) <= 8]
    final_distance = -1
    if final_goal is not None:
        final_distance = int(_distance_map(game, final_goal, cache)[carrier_pos])

    macro = own_macro(game)
    env = getattr(macro, "env", None)
    strategy = str(getattr(env, "current_strategy", "") or "")
    assignments = getattr(env, "assignment", {})
    carrier_assignment = assignments.get(carrier.name, (None, None, "MAIN"))
    carrier_side = carrier_assignment[0] if len(carrier_assignment) > 0 else None
    carrier_role = carrier_assignment[2] if len(carrier_assignment) > 2 else "MAIN"

    def same_route(ally):
        assignment = assignments.get(ally.name, (None, None, "MAIN"))
        side = assignment[0] if len(assignment) > 0 else None
        role = assignment[2] if len(assignment) > 2 else "MAIN"
        # Selling a fake and waiting with the spike are intentionally separate.
        # Formation starts after the real group rotates or executes.
        if "FAKE" in strategy and carrier_role in {"FAKE_SELL", "FAKE_WAIT"}:
            return False
        if "SPLIT" in strategy or strategy == "MID_TO_B":
            return role == carrier_role
        if strategy == "DEFAULT":
            return (carrier_role in {"MAIN_LEAD", "DEFAULT_MAIN"}
                    and role in {"MAIN_LEAD", "DEFAULT_MAIN"}
                    and side == carrier_side)
        return (side == carrier_side
                and role not in {"FAKE_SELL", "FAKE_WAIT", "SUPPORT",
                                 "OPPOSITE_SCOUT", "OPPOSITE_SCOUT_LURK",
                                 "OPPOSITE_SCOUT_DEEP", "A_SCOUT", "B_SCOUT"})

    candidates = [ally for ally in allies if same_route(ally)]
    formation_target = carrier_pos
    cursor = carrier_pos
    # A stable target three route cells ahead gives the shared Escort policy a
    # concrete formation to maintain instead of a one-tick "someone is ahead" label.
    for _ in range(FORMATION_LEAD_STEPS):
        here = int(route_distance[cursor])
        next_cells = []
        for dr, dc in CARDINAL:
            nxt = (cursor[0] + dr, cursor[1] + dc)
            if (0 <= nxt[0] < route_distance.shape[0] and 0 <= nxt[1] < route_distance.shape[1]
                    and 0 <= int(route_distance[nxt]) < here):
                next_cells.append(nxt)
        if not next_cells:
            break
        cursor = min(next_cells, key=lambda cell: (int(route_distance[cell]), cell))
        formation_target = cursor

    target_distance = _distance_map(game, formation_target, cache)
    reachable = [ally for ally in candidates
                 if int(target_distance[tuple(map(int, ally.pos))]) >= 0]
    designation_key = (strategy, carrier.name, carrier_side, carrier_role)
    saved_key = getattr(env, "_gc_screening_designation_key", None)
    saved_name = getattr(env, "_gc_screening_designated_name", None)
    tick = int(getattr(game, "battle_tick", 0))
    last_tick = int(getattr(env, "_gc_screening_last_tick", tick))
    if tick < last_tick:
        saved_key = saved_name = None
    if env is not None:
        env._gc_screening_last_tick = tick
    designated = next((ally for ally in reachable
                       if saved_key == designation_key and ally.name == saved_name), None)
    if designated is None:
        designated = min(
            reachable,
            key=lambda ally: (int(target_distance[tuple(map(int, ally.pos))]), ally.name),
            default=None,
        )
        if env is not None:
            env._gc_screening_designation_key = designation_key
            env._gc_screening_designated_name = (designated.name if designated is not None else None)
    designated_distance = (-1 if designated is None else
                           int(target_distance[tuple(map(int, designated.pos))]))
    designated_route_distance = (-1 if designated is None else
                                 int(route_distance[tuple(map(int, designated.pos))]))
    screen_ready = bool(designated is not None and designated_distance <= FORMATION_READY_RADIUS
                        and current >= 0 and 0 <= designated_route_distance <= current - 2)
    return {"route_goal": route_goal, "final_distance": final_distance,
            "ahead": ahead, "near": near, "route_distance": route_distance,
            "formation_target": formation_target, "formation_target_distance": target_distance,
            "formation_candidates": candidates, "designated": designated,
            "designated_distance": designated_distance, "screen_ready": screen_ready}


def carrier_screening_features(game, char, chars, cache, escort=False):
    """Four own-team features for learned Carrier/Escort coordination."""
    features = np.zeros(SCREENING_DIM, dtype=np.float32)
    carrier = next((c for c in chars if c.team == char.team and getattr(c, "is_alive", True)
                    and getattr(c, "has_spike", False)), None)
    if carrier is None:
        return features
    status = carrier_screening_status(game, carrier, chars, cache)
    norm = max(1, sum(game.grid.shape))
    features[0] = max(0, status["final_distance"]) / norm
    if escort:
        features[1] = float(any(c.name == char.name for c in status["ahead"]))
        features[2] = sum(c.name != char.name for c in status["ahead"]) / 4.0
        features[3] = float(navigation_intent(game, char)[2] in SCREENING_ROLES)
    else:
        features[1] = len(status["ahead"]) / 4.0
        features[2] = len(status["near"]) / 4.0
        features[3] = float(0 <= status["final_distance"] <= 24 and not status["ahead"])
    return features


def carrier_formation_features(game, char, chars, cache, escort=False):
    """Learnable state for one persistent, Macro-compatible entry screener."""
    features = np.zeros(FORMATION_DIM, dtype=np.float32)
    carrier = next((c for c in chars if c.team == char.team and getattr(c, "is_alive", True)
                    and getattr(c, "has_spike", False)), None)
    if carrier is None:
        return features
    status = carrier_screening_status(game, carrier, chars, cache)
    designated = status["designated"]
    norm = max(1, sum(game.grid.shape))
    active_approach = (FORMATION_MIN_FINAL_DISTANCE < status["final_distance"]
                       <= FORMATION_MAX_FINAL_DISTANCE)
    if escort:
        features[0] = float(designated is not None and designated.name == char.name)
        features[1] = float(status["screen_ready"])
        features[2] = (max(0, status["designated_distance"]) / norm
                       if features[0] else 0.0)
        features[3] = float(any(c.name == char.name for c in status["formation_candidates"]))
    else:
        features[0] = float(designated is not None)
        features[1] = float(status["screen_ready"])
        features[2] = max(0, status["designated_distance"]) / norm
        features[3] = float(active_approach)
    return features


def carrier_entry_sync_feature(game, char, chars, cache, navigation_idle):
    """Observable bounded-sync state; policies still choose the actual action."""
    if not getattr(char, "has_spike", False):
        return 0.0
    status = carrier_screening_status(game, char, chars, cache)
    return float(
        status["designated"] is not None
        and status["final_distance"] == ENTRY_SYNC_STAGING_DISTANCE
        and not status["screen_ready"]
        and 0 <= status["designated_distance"] <= ENTRY_SYNC_MAX_FORMATION_DISTANCE
        and int(getattr(game, "round_timer", 0)) >= ENTRY_SYNC_MIN_ROUND_TIME
        and float(navigation_idle) < ENTRY_SYNC_MAX_IDLE_FEATURE
    )


def designated_route_blocking(status, carrier):
    """Whether the designated screener occupies Carrier's next route ring."""
    designated = status.get("designated")
    if designated is None:
        return False
    route_distance = status.get("route_distance")
    if route_distance is None:
        return False
    carrier_pos = tuple(map(int, carrier.pos))
    designated_pos = tuple(map(int, designated.pos))
    current = int(route_distance[carrier_pos])
    ahead = int(route_distance[designated_pos])
    adjacent = max(abs(designated_pos[0] - carrier_pos[0]),
                   abs(designated_pos[1] - carrier_pos[1])) == 1
    return bool(adjacent and current >= 0 and 0 <= ahead < current)


def carrier_route_blocking_feature(game, char, chars, cache):
    """Own-team observation that identifies a designated route blocker."""
    carrier = next((c for c in chars if c.team == char.team and getattr(c, "is_alive", True)
                    and getattr(c, "has_spike", False)), None)
    if carrier is None:
        return 0.0
    status = carrier_screening_status(game, carrier, chars, cache)
    designated = status.get("designated")
    return float(designated is not None and designated.name == char.name
                 and designated_route_blocking(status, carrier))


def carrier_route_clearance_features(game, char, chars, cache):
    """Legal U/D/L/R moves that clear a designated Escort from Carrier's route."""
    features = np.zeros(ROUTE_CLEARANCE_DIM, dtype=np.float32)
    carrier = next((c for c in chars if c.team == char.team and getattr(c, "is_alive", True)
                    and getattr(c, "has_spike", False)), None)
    if carrier is None:
        return features
    status = carrier_screening_status(game, carrier, chars, cache)
    designated = status.get("designated")
    if (designated is None or designated.name != char.name
            or not designated_route_blocking(status, carrier)):
        return features

    grid = game.grid
    carrier_pos = tuple(map(int, carrier.pos))
    escort_pos = tuple(map(int, char.pos))
    route_distance = status["route_distance"]
    current = int(route_distance[carrier_pos])
    occupied = {tuple(map(int, other.pos)) for other in chars
                if other.name != char.name and getattr(other, "is_alive", True)}
    for i, (dr, dc) in enumerate(CARDINAL):
        nxt = (escort_pos[0] + dr, escort_pos[1] + dc)
        legal = (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                 and grid[nxt] != 1 and nxt not in occupied)
        if not legal:
            continue
        still_adjacent = max(abs(nxt[0] - carrier_pos[0]),
                             abs(nxt[1] - carrier_pos[1])) == 1
        still_ahead = current >= 0 and 0 <= int(route_distance[nxt]) < current
        features[i] = float(not (still_adjacent and still_ahead))
    return features


def carrier_safe_screen_advance_features(game, char, chars, cache):
    """Legal U/D/L/R moves toward the screen point without blocking Carrier."""
    features = np.zeros(SAFE_SCREEN_ADVANCE_DIM, dtype=np.float32)
    carrier = next((c for c in chars if c.team == char.team and getattr(c, "is_alive", True)
                    and getattr(c, "has_spike", False)), None)
    if carrier is None:
        return features
    status = carrier_screening_status(game, carrier, chars, cache)
    designated = status.get("designated")
    if (designated is None or designated.name != char.name or status["screen_ready"]
            or not (FORMATION_MIN_FINAL_DISTANCE < status["final_distance"]
                    <= FORMATION_MAX_FINAL_DISTANCE)):
        return features

    grid = game.grid
    pos = tuple(map(int, char.pos))
    carrier_pos = tuple(map(int, carrier.pos))
    target_distance = status["formation_target_distance"]
    route_distance = status["route_distance"]
    current_target = int(target_distance[pos])
    carrier_route = int(route_distance[carrier_pos])
    occupied = {tuple(map(int, other.pos)) for other in chars
                if other.name != char.name and getattr(other, "is_alive", True)}
    for i, (dr, dc) in enumerate(CARDINAL):
        nxt = (pos[0] + dr, pos[1] + dc)
        legal = (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                 and grid[nxt] != 1 and nxt not in occupied)
        if not legal or current_target < 0 or not 0 <= int(target_distance[nxt]) < current_target:
            continue
        adjacent = max(abs(nxt[0] - carrier_pos[0]), abs(nxt[1] - carrier_pos[1])) == 1
        blocks_route = (adjacent and carrier_route >= 0
                        and 0 <= int(route_distance[nxt]) < carrier_route)
        features[i] = float(not blocks_route)
    return features
