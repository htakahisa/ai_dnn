"""Add one reinforcement to learned GC positions without rebuilding the squad."""
import numpy as np
from grid_paths import distance_map
from gc_v1.map_data_search_gc import SEARCH_MAZE_STR
from ghost_champions_v2.geometry import axis_for


class DefensivePosts:
    def __init__(self, scenario):
        self.scenario = scenario
        self.setup_grid = scenario.grid.copy()
        self.setup_grid[scenario.setup != 0] = 1
        self.markers = {(r, c): int(ch) for r, row in enumerate(SEARCH_MAZE_STR.strip().splitlines())
                        for c, ch in enumerate(row) if ch in "56789"}
        self.fields = {}

    def side(self, position):
        return axis_for(position, self.scenario.grid.shape[1])

    def roles(self, assignments):
        return {name: "mid_watch" if self.side(p) == "Mid" else "guard_" + self.side(p)
                for name, p in assignments.items()}

    def reinforce_setup(self, assignments, allies, side, ticks):
        """Change at most one opposite-site player, retaining GC anchors and Mid."""
        result = {str(name): tuple(p) for name, p in assignments.items()}
        counts = {s: sum(self.side(p) == s for p in result.values()) for s in ("A", "Mid", "B")}
        other = "B" if side == "A" else "A"
        if counts[side] >= 3 or counts[other] < 2 or counts["Mid"] < 1:
            return result, None
        own = {str(c.name): c for c in allies}
        anchors = [p for p in result.values() if self.side(p) == side]
        approach_grid = self.setup_grid.copy()
        # Retained anchors become stationary before Setup ends. The added
        # player's route must not require walking through one of them.
        for p in result.values():
            approach_grid[p] = 1
        choices = []
        # A has two learned GC Setup anchors. Additional
        # support cells must stay close to a learned anchor, rather than
        # sending the squad to unrelated retake rally points.
        candidates = list(self.markers)
        candidates += [tuple(map(int, p)) for p in np.argwhere(self.setup_grid != 1)
                       if self.side(p) == side and tuple(p) not in self.markers]
        for p in candidates:
            if p in result.values() or self.side(p) != side or self.setup_grid[p] == 1:
                continue
            if p not in self.fields:
                self.fields[p] = distance_map(self.setup_grid, p)
            field = self.fields[p]
            support = min((int(field[q]) for q in anchors if field[q] >= 0), default=10000)
            if p not in self.markers and not 2 <= support <= 3:
                continue
            approach = distance_map(approach_grid, p)
            for name, old in result.items():
                if self.side(old) != other or name not in own:
                    continue
                distance = int(approach[tuple(own[name].pos)])
                if 0 <= distance <= ticks - 2:
                    choices.append((support, distance, p not in self.markers, int(field[old]), name, p))
        if not choices:
            return result, None
        *_, name, p = min(choices)
        result[name] = p
        return result, name

    def keep_search_roles(self, search, setup_goals, grid):
        """Hand off Setup roles without overwriting any lower-layer action.

        GC Search independently redraws five patrol posts at LIVE. Retain
        every draw on the intended side; only redraw mismatched roles so
        the historical 3/1/1 allocation and Mid survive that transition.
        """
        assigned = search._assigned_positions
        keep = {name for name, points in assigned.items() if points and
                (name not in setup_goals or self.side(points[0]) == self.side(setup_goals[name]))}
        used = {tuple(p) for name in keep for p in assigned[name]}
        for name in sorted(set(assigned).intersection(setup_goals) - keep):
            side = self.side(setup_goals[name])
            field = distance_map(grid, setup_goals[name])
            candidates = [p for p in self.markers if p not in used and self.side(p) == side and field[p] >= 0]
            if not candidates:
                continue
            goal = min(candidates, key=lambda p: (int(field[p]), p))
            assigned[name] = [goal]
            search._assigned_dist_maps[name] = distance_map(grid, goal)
            search._assigned_markers[name] = self.markers[goal]
            used.add(goal)
        return {name: tuple(points[0]) for name, points in assigned.items() if points}
