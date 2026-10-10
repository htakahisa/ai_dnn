"""Public, persistent rally assignments and coordinated site entry."""
import math
from grid_paths import distance_map
from game_core import DEFUSE_REQUIRED_TICKS

RETAKE_VERSION = "living_defuser_opponent_learning_v4"
MAX_RALLY_WAIT_TICKS = 6


def retake_layout(scenario):
    return dict(rally=scenario.rally_points, entries=scenario.retake_entries, utility=scenario.retake_utility)


class RetakeAssembly:
    def __init__(self, scenario):
        self.scenario = scenario
        self.assigned = {}
        self.ready = set()
        self.entered = set()
        self.launched = False
        self.side = None
        self.started_tick = None

    def goals(self, snapshot):
        side = self.scenario.site_of(snapshot.spike_planted)
        if self.side is not None and self.side != side:
            self.assigned.clear()
            self.ready.clear()
            self.entered.clear()
            self.launched = False
        self.side = side
        if self.started_tick is None:
            self.started_tick = snapshot.tick
        allies = sorted((a for a in snapshot.allies if a.alive), key=lambda a: (a.position, a.ability_name))
        alive = {a.slot for a in allies}
        self.assigned = {slot: point for slot, point in self.assigned.items() if slot in alive}
        groups = self.scenario.rally_points[side]
        group_of = {p: marker for marker, cells in groups.items() for p in cells}
        maps = {p: distance_map(self.scenario.grid, p) for p in group_of}
        capacity = max(1, math.ceil(len(allies) / len(groups)))
        for ally in allies:
            if ally.slot in self.assigned:
                continue
            counts = {marker: sum(group_of[p] == marker for p in self.assigned.values()) for marker in groups}
            candidates = [p for p in maps if p not in self.assigned.values() and maps[p][ally.position] >= 0]
            balanced = [p for p in candidates if counts[group_of[p]] < capacity]
            self.assigned[ally.slot] = min(balanced or candidates, key=lambda p: (maps[p][ally.position], p))
        spike_routes = distance_map(self.scenario.grid, snapshot.spike_planted)
        budget = snapshot.detonate_timer - DEFUSE_REQUIRED_TICKS - 3
        eligible = set()
        self.ready.clear()
        for ally in allies:
            point = self.assigned[ally.slot]
            group = groups[group_of[point]]
            if any(0 <= maps[p][ally.position] <= 2 for p in group):
                self.ready.add(ally.slot)
            eta = maps[point][ally.position]
            if eta >= 0 and spike_routes[point] >= 0 and eta + spike_routes[point] <= budget:
                eligible.add(ally.slot)
        if (len(allies) <= 1 or not eligible or eligible <= self.ready
                or snapshot.tick-self.started_tick >= MAX_RALLY_WAIT_TICKS and len(self.ready & alive) >= 2
                or any(0 <= spike_routes[a.position] <= 1 for a in allies)
                or any(spike_routes[self.assigned[s]] >= budget for s in eligible)):
            self.launched = True
        if not self.launched:
            return dict(self.assigned)
        result = {}
        for ally in allies:
            marker = group_of[self.assigned[ally.slot]]
            entries = self.scenario.retake_entries[side].get(marker, ())
            if entries:
                routes = {p: distance_map(self.scenario.grid, p) for p in entries}
                reachable = [p for p in entries if routes[p][ally.position] >= 0]
                if any(0 <= routes[p][ally.position] <= 1 for p in reachable):
                    self.entered.add(ally.slot)
                if reachable and ally.slot not in self.entered:
                    result[ally.slot] = min(reachable, key=lambda p: routes[p][ally.position])
                    continue
            result[ally.slot] = snapshot.spike_planted
        return result
