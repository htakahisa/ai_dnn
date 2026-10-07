"""Persistent assembly goals; learned policy still chooses movement and combat."""

import math
import numpy as np

from concon_v1.co1_attacker_common import bfs_distance_map
from concon_v1.co1_retake_navigation import assembly_navigation
from concon_v1.co1_retake_config import COORDINATION_VERSION
from game_core import DEFUSE_REQUIRED_TICKS

COMBAT_RESERVE = 3
FORWARD_SLACK = 3


class RetakeAssembly:
    def __init__(self, scenario, version=COORDINATION_VERSION):
        self.scenario = scenario
        self.version = version
        if version == 2:
            self.front, self.routes = assembly_navigation(scenario, version=2)
        elif version in (1, 3):
            # Paired entry coordination uses ordinary assembly distances.
            self.front = np.zeros(scenario.grid.shape, dtype=bool)
            self.routes = {point: bfs_distance_map(scenario.grid, point) for point in scenario.rally_points}
        else:
            raise ValueError(f"unsupported retake coordination version: {version}")
        self.reset()

    def reset(self):
        self.assigned = {}
        self.visited = set()
        self.ready = set()
        self.reached = set()
        self.launched = False
        self.entered = set()
        self.tick = None
        self.snapshot = None

    def update(self, allies, spike, remaining, tick):
        # All actors use the same release decision even though the engine asks
        # for their actions sequentially and IQ perturbs their ally coordinates.
        if self.tick == tick:
            return self.snapshot
        self.tick = tick
        groups = self.scenario.rally_groups
        alive = {ally.name: ally for ally in allies}
        self.assigned = {name: point for name, point in self.assigned.items() if name in alive}
        capacity = max(1, math.ceil(len(allies) / len(groups)))
        group_of = {point: i for i, group in enumerate(groups) for point in group}
        spike_routes = bfs_distance_map(self.scenario.grid, spike)
        occupied = set(self.assigned.values())
        for ally in sorted(allies, key=lambda a: a.name):
            if ally.name in self.assigned:
                continue
            counts = [sum(group_of[p] == i for p in self.assigned.values()) for i in range(len(groups))]
            choices = [p for p in self.routes if p not in occupied and self.routes[p][tuple(ally.pos)] >= 0]
            balanced = [p for p in choices if counts[group_of[p]] < capacity]
            if balanced:
                choices = balanced
            if choices:
                point = min(choices, key=lambda p: (self.routes[p][tuple(ally.pos)], counts[group_of[p]], p))
                self.assigned[ally.name] = point
                occupied.add(point)
        eta = {name: int(self.routes[p][tuple(alive[name].pos)]) for name, p in self.assigned.items()}
        # A marks an entrance area, not a mandatory individual stopping cell.
        # IQ can move an ally estimate one row AND one column away from A:
        # accept that diagonal only when it is also reachable in two steps.
        # Keep the route check so nearby cells across a wall do not count.
        # Arrival remains latched across estimates and optional advances.
        for name, point in self.assigned.items():
            position = tuple(alive[name].pos)
            entrance = groups[group_of[point]]
            if any(max(abs(position[0] - p[0]), abs(position[1] - p[1])) <= 1
                   and 0 <= self.routes[p][position] <= 2 for p in entrance):
                self.ready.add(name)
                self.reached.add((name, self.assigned[name]))
                self.visited.add((name, group_of[self.assigned[name]]))
        budget = remaining - DEFUSE_REQUIRED_TICKS - COMBAT_RESERVE
        eligible = {name for name, p in self.assigned.items()
                    if eta[name] >= 0 and spike_routes[p] >= 0 and eta[name] + spike_routes[p] <= budget}
        at_site = any(0 <= spike_routes[tuple(a.pos)] <= 1 for a in allies)
        ready = eligible <= self.ready
        deadline = any(spike_routes[self.assigned[name]] >= budget for name in eligible)
        if len(allies) <= 1 or at_site or not eligible or ready or deadline:
            self.launched = True
        if not self.launched and self.version != 3:
            # Commit at most once to each entrance. Followers retain their A;
            # nobody chases a moving leader's newly selected assembly point.
            for name in sorted(eligible & self.ready):
                current = self.assigned[name]
                follower_eta = max((eta[n] for n in eligible - self.ready), default=0)
                if follower_eta <= 0:
                    continue
                source = tuple(alive[name].pos)
                counts = [sum(group_of[p] == i for p in self.assigned.values()) for i in range(len(groups))]
                choices = [p for p in self.routes if p not in self.assigned.values()
                           and (name, group_of[p]) not in self.visited
                           and counts[group_of[p]] < capacity
                           and 0 <= self.routes[p][source] <= follower_eta + FORWARD_SLACK
                           and 0 <= spike_routes[p] < spike_routes[current]
                           and self.routes[p][source] + spike_routes[p] <= budget]
                if choices:
                    self.assigned[name] = min(choices, key=lambda p: (spike_routes[p], self.routes[p][source], p))
                    self.visited.add((name, group_of[self.assigned[name]]))
                    # Prior readiness remains latched: the leader's optional
                    # advance must never extend the followers' release time.
        self.snapshot = dict(assigned=dict(self.assigned), eligible=eligible,
                             near=len(eligible & self.ready), waiting=not self.launched,
                             urgent=len(eligible) < len(allies) or deadline)
        return self.snapshot
