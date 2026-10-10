"""Plant-only public rally planning and traffic-aware training inputs."""
import numpy as np
from collections import deque
from functools import lru_cache
from grid_paths import distance_map
from game_core import PLANT_REQUIRED_TICKS
from frc_v1.actions import MOVE_STEPS

PREPARE_STEPS = 18
MAX_ENTRY_WAIT = 6
MAX_FLANK_EXTRA_STEPS = 10
MAX_TRAFFIC_DETOUR_STEPS = 4
SITE_SWITCH_MARGIN = .10
SITE_SWITCH_TRAVEL_COST = .01
ENTRY_READY_DISTANCE = 1
ENTRY_RESERVE = 8
ENTRY_FEATURES = 17  # five traffic reachability/progress pairs; seven rally/mission fields.
MOVES = ('STAY', 'N', 'E', 'S', 'W')


def site_entry_schema():
    return dict(version=2, prepare_steps=PREPARE_STEPS, max_wait=MAX_ENTRY_WAIT,
                flank_extra=MAX_FLANK_EXTRA_STEPS, ready_distance=ENTRY_READY_DISTANCE,
                reserve=ENTRY_RESERVE, features=ENTRY_FEATURES,
                wait_limit='elapsed_since_first_ready_without_quorum',
                spike_recovery='release_rally_when_carrier_changes',
                teacher_detour_limit=MAX_TRAFFIC_DETOUR_STEPS,
                progress_reward='fixed_nearest_plant_distance_no_dynamic_traffic_credit',
                site_switch_margin=SITE_SWITCH_MARGIN, site_switch_travel_cost=SITE_SWITCH_TRAVEL_COST,
                tactical_wait='not_a_route_stall',
                staging='distinct_outside_cells_two_public_map_entries',
                traffic='public_allies_blocked_goal_open', selection='learned_legal_actions')


def teacher_distances(static, traffic, position):
    """A moving teammate is not grounds for a lap around the whole map."""
    if (traffic is None or traffic[position] < 0 or static[position] < 0
            or traffic[position]-static[position] > MAX_TRAFFIC_DETOUR_STEPS):
        return static
    return traffic


def mission_progress(scenario, before, after):
    """A fixed public-map potential: round trips cannot earn net progress.

    Use the nearest plant cell on either site, not moving rally targets or
    changing ally blockers. A necessary tactical detour can still win through
    the plant/survival outcome rewards; it is not itself mission progress.
    """
    def distance(position):
        return min((int(field[tuple(position)]) for field in scenario.site_dist.values()
                    if field[tuple(position)] >= 0), default=-1)
    old, new = distance(before), distance(after)
    return float(old-new) if old >= 0 and new >= 0 else 0.


def traffic_distances(snapshot, ally, goal, blocked=()):
    board = np.asarray(snapshot.grid)
    obstacles = set(blocked)
    obstacles.update(other.position for other in snapshot.allies
                     if other.alive and other.slot != ally.slot and other.position != goal)
    obstacles.discard(ally.position)
    obstacles.discard(goal)
    # An occupied waypoint may be reached after its teammate moves. Legal
    # masks still prohibit entering an occupied/reserved destination this tick.
    shape = board.shape
    columns = shape[1]
    return _traffic_map(shape, (board == 1).tobytes(), goal[0]*columns+goal[1],
                        frozenset(r*columns+c for r,c in obstacles)).copy()


@lru_cache(maxsize=16)
def _walk_graph(shape, walls):
    rows, columns = shape
    graph = []
    for index, wall in enumerate(walls):
        if wall:
            graph.append(())
            continue
        row, column = divmod(index, columns)
        neighbours = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row+dr, column+dc
            if 0 <= r < rows and 0 <= c < columns and not walls[r*columns+c]:
                neighbours.append(r*columns+c)
        graph.append(tuple(neighbours))
    return tuple(graph)


@lru_cache(maxsize=256)
def _traffic_map(shape, walls, goal, blocked):
    """Integer graph BFS avoids NumPy scalar work and the static path cache."""
    values = [-1] * (shape[0]*shape[1])
    if not walls[goal]:
        graph = _walk_graph(shape, walls)
        values[goal] = 0
        queue = deque([goal])
        while queue:
            index = queue.popleft()
            value = values[index]+1
            for neighbour in graph[index]:
                if values[neighbour] < 0 and neighbour not in blocked:
                    values[neighbour] = value
                    queue.append(neighbour)
    result = np.asarray(values, np.int32).reshape(shape)
    result.setflags(write=False)
    return result


def site_entry_features(snapshot, ally, goal, traffic, state):
    result = []
    old = traffic[ally.position]
    for kind in MOVES:
        dr, dc = MOVE_STEPS.get(kind, (0, 0))
        p = ally.position[0] + dr, ally.position[1] + dc
        valid = (0 <= p[0] < traffic.shape[0] and 0 <= p[1] < traffic.shape[1]
                 and traffic[p] >= 0)
        result += [float(valid), float(np.clip(old-traffic[p], -1, 1)) if valid and old >= 0 else 0.]
    state = state or {}
    ready = state.get('ready', set())
    alive = max(1, sum(a.alive for a in snapshot.allies))
    travel = max(0, traffic[ally.position])
    slack = snapshot.round_timer - travel - max(0, PLANT_REQUIRED_TICKS - ally.plant_progress)
    result += [float(state.get('active', False)), float(state.get('launched', False)),
               float(ally.slot in ready), len(ready)/alive,
               float(ally.slot in state.get('flank', set())),
               float(np.clip(slack/100, -1, 1)), float(snapshot.round_timer <= travel + PLANT_REQUIRED_TICKS + ENTRY_RESERVE)]
    return np.asarray(result, np.float32)


class PlantSiteAssembly:
    """Assign public goals, never override or force learned actions."""
    def __init__(self, scenario):
        self.scenario = scenario
        self.side = None
        self.assigned = {}
        self.entries = {}
        self.flank = set()
        self.started = None
        self.launched = False
        self.entered = set()
        self.carrier_slot = None

    def observe(self, snapshot, route):
        alive = [a for a in snapshot.allies if a.alive]
        if hasattr(snapshot, 'enemies') and not any(e.alive for e in snapshot.enemies):
            return None
        holder = next((a for a in alive if a.has_spike), None)
        if self.side != route.site:
            self.__init__(self.scenario)
            self.side = route.site
        if not holder or holder.plant_progress or self.scenario.grid[holder.position] == 2:
            return None
        if self.carrier_slot is not None and self.carrier_slot != holder.slot:
            self.launched = True  # The retriever must not wait at a support player's old rally goal.
            self.carrier_slot = holder.slot
        if not self.assigned:
            to_site = self.scenario.site_dist[route.site]
            if to_site[holder.position] < 0 or to_site[holder.position] > PREPARE_STEPS:
                return None
            index = next((i for i,p in enumerate(route.cells) if self.scenario.grid[p] == 2), None)
            if index is None or index == 0:
                return None
            primary = route.cells[index-1], route.cells[index]
            board = self.scenario.grid.copy()
            board[board == 2] = 1
            primary_map = distance_map(board, primary[0])
            alternatives = []
            for inside in self.scenario.sites[route.site]:
                for outside in self.scenario.neighbors(inside):
                    if board[outside] == 1 or abs(outside[0]-primary[0][0])+abs(outside[1]-primary[0][1]) < 3:
                        continue
                    field = distance_map(board, outside)
                    players = sorted((a for a in alive if not a.has_spike and field[a.position] >= 0),
                                     key=lambda a: (field[a.position], a.slot))[:2]
                    if not players:
                        continue
                    eta = max(field[a.position] for a in players)
                    main_eta = primary_map[holder.position]
                    if main_eta >= 0 and eta <= main_eta + MAX_FLANK_EXTRA_STEPS and eta + PLANT_REQUIRED_TICKS + ENTRY_RESERVE < snapshot.round_timer:
                        alternatives.append((eta, outside, inside, players))
            alternate = min(alternatives, key=lambda row: row[:3]) if alternatives else None
            self.flank = {a.slot for a in alternate[3]} if alternate else set()
            self.carrier_slot = holder.slot
            used = set()
            for a in sorted(alive, key=lambda a: (not a.has_spike, a.slot)):
                outside, inside = (alternate[1], alternate[2]) if a.slot in self.flank else primary
                near = distance_map(board, outside)
                own = distance_map(board, a.position)
                candidates = [tuple(map(int,p)) for p in np.argwhere((near >= 0) & (near <= 2) & (own >= 0))
                              if tuple(p) not in used]
                if not candidates:
                    continue
                point = min(candidates, key=lambda p: (near[p], own[p], p))
                self.assigned[a.slot] = point
                self.entries[a.slot] = tuple(map(int, inside))
                used.add(point)
        slots = {a.slot for a in alive}
        self.flank &= slots
        ready = {a.slot for a in alive if a.slot in self.assigned
                 and 0 <= distance_map(self.scenario.grid, self.assigned[a.slot])[a.position] <= ENTRY_READY_DISTANCE}
        if ready and self.started is None:
            self.started = snapshot.tick
        elapsed = snapshot.tick-self.started if self.started is not None else 0
        if (slots <= ready or self.started is not None and elapsed >= MAX_ENTRY_WAIT
                or snapshot.round_timer <= PLANT_REQUIRED_TICKS + ENTRY_RESERVE
                or snapshot.round_timer <= self.scenario.site_dist[route.site][holder.position] + PLANT_REQUIRED_TICKS + ENTRY_RESERVE):
            self.launched = True
        goals = {}
        for a in alive:
            if self.scenario.grid[a.position] == 2:
                self.entered.add(a.slot)
            if a.slot in self.assigned and a.slot not in self.entered:
                goals[a.slot] = self.entries[a.slot] if self.launched else self.assigned[a.slot]
        return dict(active=True, goals=goals, ready=ready, main=slots-self.flank,
                    flank=set(self.flank), all_ready=slots <= ready,
                    wait=not self.launched, elapsed=elapsed, launched=self.launched)
