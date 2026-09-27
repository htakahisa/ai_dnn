"""Committed dropped-spike recovery and the complete pickup-to-plant route.

Only perceived occupants, the observed drop location, public time and static
map geometry are used. Enemy visibility cannot reverse the recovery route.
"""

from collections import deque

import numpy as np

from .opponent_site_gc import attack_plant_cells

from game_core import PLANT_REQUIRED_TICKS

try:
    from .positioning_gc import set_team_plant_target
except ImportError:
    from positioning_gc import set_team_plant_target

CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
RECOVERY_TIME_MARGIN = 10  # Pickup tick, perception search, movement and timer rounding.


def _pos(char):
    return tuple(map(int, char.pos))


def _path(grid, start, goal, occupied=()):
    start, goal = tuple(start), tuple(goal)
    blocked = set(occupied) - {start}
    if goal in blocked:
        return None
    height, width = grid.shape
    if not (0 <= goal[0] < height and 0 <= goal[1] < width) or grid[goal] == 1:
        return None
    parents = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            route = [cell]
            while parents[route[-1]] is not None:
                route.append(parents[route[-1]])
            return route[::-1]
        for dr, dc in CARDINAL:
            nxt = (cell[0] + dr, cell[1] + dc)
            if (0 <= nxt[0] < height and 0 <= nxt[1] < width
                    and grid[nxt] != 1 and nxt not in blocked and nxt not in parents):
                parents[nxt] = cell
                queue.append(nxt)
    return None


def _closest_plant(grid, origin, allowed=None):
    queue = deque([(tuple(origin), 0)])
    visited = {tuple(origin)}
    while queue:
        cell, distance = queue.popleft()
        if grid[cell] == 2 and (allowed is None or cell in allowed):
            return distance, cell
        for dr, dc in CARDINAL:
            nxt = (cell[0] + dr, cell[1] + dc)
            if (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                    and grid[nxt] != 1 and nxt not in visited):
                visited.add(nxt)
                queue.append((nxt, distance + 1))
    return None


class SpikeRecoveryCoordinator:
    def __init__(self):
        self.reset_round()

    def reset_round(self):
        self.plan = None
        self.last_status = None

    def _new_drop(self, grid, attackers, observed):
        routes = []
        for attacker in attackers:
            route = _path(grid, _pos(attacker), observed)
            if route is not None:
                routes.append((len(route) - 1, attacker.name))
        if not routes:
            self.plan = None
            return
        self.plan = {
            "phase": "RECOVER", "retriever": min(routes)[1],
            "estimate": observed, "target": observed, "visited": set(),
            "observations": {observed}, "plant": None,
            "pickup_attempt_at": None,
            "search_radius": 1,
        }

    def _search_target(self, grid, retriever):
        plan = self.plan
        current = _pos(retriever)
        if current != plan["target"]:
            return plan["target"]
        if plan["pickup_attempt_at"] != current:
            # Starting on the reported drop still needs one engine pickup tick.
            plan["pickup_attempt_at"] = current
            return current
        # The engine tried automatic pickup after the last move. If no holder
        # exists now, this perceived location was inaccurate; search nearby.
        plan["visited"].add(current)
        centers = plan["observations"]
        cells = set()
        radius = plan["search_radius"]
        for r, c in centers:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    cell = (r + dr, c + dc)
                    if (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]
                            and grid[cell] != 1):
                        cells.add(cell)
        ranked = []
        for cell in cells - plan["visited"]:
            route = _path(grid, current, cell)
            if route is not None:
                ranked.append((len(route) - 1, cell not in centers, cell))
        if ranked:
            plan["target"] = min(ranked)[2]
        elif radius < 3:
            # A noisy coordinate corrected away from a wall can be more than
            # one cell away. Expand the local search instead of standing still.
            plan["search_radius"] += 1
            return self._search_target(grid, retriever)
        else:
            if not any(cell != current and _path(grid, current, cell) is not None
                       for cell in cells):
                return current
            plan["visited"] = {current}
            return self._search_target(grid, retriever)
        return plan["target"]

    @staticmethod
    def _yield_result(char, actor, target, grid, chars):
        """Clear a teammate off the committed route without assigning a new retriever."""
        route = _path(grid, _pos(actor), target)
        if route is None or _pos(char) not in route[1:]:
            return None
        occupied = {_pos(other) for other in chars
                    if other.name != char.name and getattr(other, "is_alive", True)}
        candidates = []
        for dr, dc in CARDINAL:
            nxt = (_pos(char)[0] + dr, _pos(char)[1] + dc)
            if (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                    and grid[nxt] != 1 and nxt not in occupied):
                candidates.append((nxt in route, abs(nxt[0] - target[0])
                                   + abs(nxt[1] - target[1]), nxt))
        if candidates:
            if int(getattr(char, "move_steps_per_tick", 1)) > 1:
                return list(min(candidates)[2]), "MOVE", {"move_step_limit": 1}
            return list(min(candidates)[2]), "MOVE"
        return list(_pos(char)), "MOVE"

    def decide_move(self, char, game_state, game=None):
        if game_state.get("is_planted"):
            self.reset_round()
            return None
        if not getattr(char, "is_alive", True):
            return None
        chars = game_state.get("chars", [])
        attackers = [other for other in chars if other.team == "A"
                     and getattr(other, "is_alive", True)]
        holder = next((other for other in attackers if getattr(other, "has_spike", False)), None)
        observed = game_state.get("spike_pos")
        if self.plan is None and (holder is not None or observed is None):
            return None
        if "grid" not in game_state:
            return None
        grid = np.asarray(game_state["grid"])
        plant_cells = set(attack_plant_cells(game, grid))
        if holder is None:
            if observed is None:
                return None
            observed = tuple(map(int, observed))
            if self.plan is None or self.plan["phase"] != "RECOVER":
                self._new_drop(grid, attackers, observed)
            if self.plan is None:
                return None
            plan = self.plan
            retriever = next((other for other in attackers if other.name == plan["retriever"]), None)
            if retriever is None:
                self._new_drop(grid, attackers, observed)
                if self.plan is None:
                    return None
                plan = self.plan
                retriever = next(other for other in attackers if other.name == plan["retriever"])
            # Different viewers/ticks may report a one-cell IQ error. Keep the
            # current approach fixed and remember nearby reports for searching.
            if max(abs(observed[i] - plan["estimate"][i]) for i in (0, 1)) <= 3:
                plan["observations"].add(observed)
            actor = retriever
            target = (self._search_target(grid, actor) if char.name == actor.name
                      else plan["target"])
        else:
            plan = self.plan
            if (plan["phase"] == "RECOVER" or plan["retriever"] != holder.name
                    or plan["plant"] not in plant_cells):
                nearest = _closest_plant(grid, _pos(holder), plant_cells)
                if nearest is None:
                    return None
                plan.update(phase="PLANT", retriever=holder.name, plant=nearest[1])
            actor = holder
            target = _pos(holder) if _pos(holder) in plant_cells else plan["plant"]
            if game is not None:
                set_team_plant_target(game, target)
            game_state["target_plant_pos"] = target

        route = _path(grid, _pos(actor), target)
        plant_route = _closest_plant(grid, target, plant_cells) if holder is None else (0, target)
        distance = len(route) - 1 if route is not None else None
        required = (distance + plant_route[0] + PLANT_REQUIRED_TICKS + RECOVERY_TIME_MARGIN
                    if distance is not None and plant_route is not None else None)
        remaining = int(game_state.get("round_timer", getattr(game, "round_timer", 180)))
        self.last_status = {
            "phase": plan["phase"], "retriever": actor.name, "target": target,
            "remaining": remaining, "required_ticks": required,
            "urgent": required is not None and remaining <= required,
        }
        if char.name != actor.name:
            return self._yield_result(char, actor, target, grid, chars)
        if holder is not None and _pos(char) in plant_cells:
            return list(_pos(char)), "PLANT"
        occupied = {_pos(other) for other in chars if other.name != char.name
                    and getattr(other, "is_alive", True)}
        route = _path(grid, _pos(char), target, occupied)
        if route is None and target in occupied:
            # A camper/ally on the spike is not a reason to freeze far away:
            # approach a free adjacent cell, then combat or yielding clears it.
            approaches = []
            for dr, dc in CARDINAL:
                candidate = (target[0] + dr, target[1] + dc)
                approach = _path(grid, _pos(char), candidate, occupied)
                if approach is not None:
                    approaches.append(approach)
            route = min(approaches, key=len) if approaches else None
        if route is not None and len(route) > 1:
            if holder is None and route[1] == target:
                plan["pickup_attempt_at"] = target
            if int(getattr(char, "move_steps_per_tick", 1)) > 1:
                # Precise recovery/search/plant routes must stop at corners and
                # pickups, not extend the chosen step through the objective.
                return list(route[1]), "MOVE", {"move_step_limit": 1}
            return list(route[1]), "MOVE"
        return list(_pos(char)), "MOVE"
