"""Public entry timing and shared firing lanes for the plant learner."""
import math
from game_core import FACING_VECTORS
from grid_paths import distance_map
from grid_lines import line_cells

ENTRY_READY_DISTANCE = 1
ENTRY_MAX_WAIT_TICKS = 6
ENTRY_TIME_RESERVE = 12
ENTRY_FEATURES = 12


def coordination_schema():
    return {"version": 1, "features": ENTRY_FEATURES, "ready_distance": ENTRY_READY_DISTANCE,
            "max_wait": ENTRY_MAX_WAIT_TICKS, "time_reserve": ENTRY_TIME_RESERVE,
            "information": "public_allies_sightings_effects_only"}


def firing_lanes(scenario, snapshot, target):
    """Only allies facing a currently public enemy count as ready support."""
    lanes = []
    smoke = set(snapshot.smoke_cells)
    for ally in snapshot.allies:
        if not ally.alive or ally.blind or ally.position == target:
            continue
        line = line_cells(ally.position, target)
        occupied = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
        dx, dy = FACING_VECTORS[ally.facing]
        dr, dc = target[0]-ally.position[0], target[1]-ally.position[1]
        if (dc*dx+dr*dy < 0 or not scenario.clear(ally.position, target)
                or occupied.intersection(line[1:-1]) or len(line) > 2 and smoke.intersection(line)):
            continue
        lanes.append(ally)
    return lanes


def shared_fire_score(scenario, snapshot, targets=None):
    targets = tuple(dict.fromkeys(targets if targets is not None else (s.position for s in snapshot.sightings)))
    values = []
    for target in targets:
        lanes = firing_lanes(scenario, snapshot, target)
        angle = 0.
        for i, a in enumerate(lanes):
            ar, ac = a.position[0]-target[0], a.position[1]-target[1]
            for b in lanes[i+1:]:
                br, bc = b.position[0]-target[0], b.position[1]-target[1]
                cosine = (ar*br+ac*bc)/(math.hypot(ar, ac)*math.hypot(br, bc))
                angle = max(angle, (1-cosine)/2)
        values.append(min(1., max(0, len(lanes)-1)/2) * angle)
    return sum(values)/len(values) if values else 0.


class EntryCoordinator:
    def __init__(self, scenario):
        self.scenario = scenario
        self.wait_started = {}

    def observe(self, snapshot, route, flanks):
        index = next((i for i,p in enumerate(route.cells) if self.scenario.grid[p] == 2), len(route.cells)-1)
        primary = route.cells[max(0,index-1)]
        goals = {a.slot: flanks[a.slot][0] if a.slot in flanks else primary
                 for a in snapshot.allies if a.alive}
        distances = {slot: distance_map(self.scenario.grid, goal) for slot,goal in goals.items()}
        ready = {a.slot for a in snapshot.allies if a.alive
                 and 0 <= distances[a.slot][a.position] <= ENTRY_READY_DISTANCE}
        main = set(goals)-set(flanks)
        flank = set(goals).intersection(flanks)
        all_ready = bool(main.intersection(ready)) and bool(flank.intersection(ready))
        key = route.site, tuple(sorted(flank))
        if main.intersection(ready) and flank and key not in self.wait_started:
            self.wait_started[key] = snapshot.tick
        elapsed = snapshot.tick-self.wait_started.get(key, snapshot.tick)
        can_join = all(0 <= distances[a.slot][a.position] <= ENTRY_MAX_WAIT_TICKS
                       for a in snapshot.allies if a.alive and a.slot in flank)
        wait = (bool(main.intersection(ready)) and bool(flank) and not all_ready and can_join
                and elapsed < ENTRY_MAX_WAIT_TICKS and snapshot.round_timer > ENTRY_TIME_RESERVE)
        return {"ready": ready, "main": main, "flank": flank, "wait": wait,
                "elapsed": elapsed, "all_ready": all_ready}


def entry_features(scenario, snapshot, ally, destination, impact, coordination=None):
    state = coordination or {"ready": set(), "main": set(), "flank": set(), "wait": False,
                             "elapsed": 0, "all_ready": False}
    alive = max(1, sum(a.alive for a in snapshot.allies))
    lane_counts = [len(firing_lanes(scenario, snapshot, s.position)) for s in snapshot.sightings]
    smoke = set(snapshot.smoke_cells)
    def exposed(p):
        return impact is not None and scenario.clear(p, impact) and not smoke.intersection(line_cells(p,impact))
    return [float(impact is not None), float(exposed(ally.position)), float(exposed(destination)),
            float(ally.slot in state['ready']), len(state['ready'])/alive,
            len(state['ready'].intersection(state['main']))/max(1,len(state['main'])),
            len(state['ready'].intersection(state['flank']))/max(1,len(state['flank'])),
            float(bool(state['flank'])), float(state['wait']),
            min(1.,state['elapsed']/ENTRY_MAX_WAIT_TICKS),
            min(3,max(lane_counts,default=0))/3, shared_fire_score(scenario,snapshot)]
