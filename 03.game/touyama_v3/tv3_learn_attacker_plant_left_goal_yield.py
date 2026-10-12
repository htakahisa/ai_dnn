"""Public left-site goal repair candidate; learned actions remain in control."""
from collections import deque
from grid_paths import distance_map
from game_core import FACING_VECTORS
from touyama_v3.tv3_attacker_site_entry import traffic_distances
from touyama_v3.tv3_learn_attacker_recon_cover import incoming_clear
from touyama_v3.tv3_learn_attacker_fire_support import shot_clear

# 通常設定：同じ入口占有が続いたときだけ、近傍へ目標を再配置する。
BLOCKED_TICKS = 3
YIELD_MAX_STEPS = 2


def left_yield_proposal(scenario, snapshot, goals, threats=()):
    """Find a stopped ally downstream of the carrier's occupancy dependency."""
    allies = {a.slot: a for a in snapshot.allies if a.alive}
    holder = next((a for a in allies.values() if a.has_spike), None)
    if holder is None or holder.plant_progress:
        return None
    entry = goals.get(holder.slot)
    if entry not in scenario.sites['L'] or entry == holder.position:
        return None
    occupied = {a.position: a for a in allies.values()}
    first = occupied.get(entry)
    if first is None or first.slot == holder.slot:
        return None
    # Follow occupied goals and occupied shortest-path next steps. Moving just
    # the entrance blocker is insufficient when its own exit is also occupied.
    queue, visited = deque([first.slot]), set()
    blockers = []
    while queue:
        slot = queue.popleft()
        if slot in visited or slot == holder.slot:
            continue
        visited.add(slot)
        ally = allies[slot]
        blockers.append(ally)
        goal = goals.get(slot)
        if goal is None:
            continue
        field = distance_map(scenario.grid, goal)
        cells = [goal]
        if field[ally.position] > 0:
            cells += [p for p in scenario.neighbors(ally.position)
                      if 0 <= field[p] < field[ally.position]]
        for cell in cells:
            other = occupied.get(cell)
            if other is not None and other.slot != slot:
                queue.append(other.slot)
    forbidden = set(occupied) | {g for g in goals.values() if g is not None}
    choices = []
    for ally in blockers:
        if (ally.blind or ally.plant_progress or ally.movement_disabled
                or ally.warping or ally.ramp_blocked):
            continue
        dx, dy = FACING_VECTORS[ally.facing]
        if any((t.position[1]-ally.position[1])*dx+(t.position[0]-ally.position[0])*dy >= 0
               and shot_clear(scenario,snapshot,ally.position,t.position,ally.slot) for t in threats):
            continue  # Preserve an ally already covering a public target.
        old_risk = sum(incoming_clear(scenario, snapshot, ally, ally.position, t) for t in threats)
        # Site floors can all be occupied. Nearby floor outside the site is
        # also a valid place to yield, provided it is not another goal/site.
        for destination, _ in scenario.local(ally.position, YIELD_MAX_STEPS):
            destination = tuple(map(int, destination))
            if scenario.site_of(destination) == 'R':
                continue
            if destination in forbidden:
                continue
            field = traffic_distances(snapshot, ally, destination)
            steps = int(field[ally.position])
            if not 0 < steps <= YIELD_MAX_STEPS:
                continue
            # Every shortest first step admitted for this goal must preserve
            # public exposure; do not trade an avoidable jam for an exposed exit.
            next_steps = [p for p in scenario.neighbors(ally.position)
                          if 0 <= field[p] < steps and p not in occupied]
            if not next_steps or any(sum(incoming_clear(scenario, snapshot, ally, p, t)
                                         for t in threats) > old_risk for p in next_steps):
                continue
            risk = sum(incoming_clear(scenario, snapshot, ally, destination, t) for t in threats)
            if risk > old_risk:
                continue
            choices.append((ally.position != goals.get(ally.slot), steps, risk, ally.slot, destination))
    if not choices:
        return None
    _, _, _, slot, destination = min(choices)
    return dict(slot=int(slot), goal=destination, entry=tuple(map(int,entry)), carrier_slot=int(holder.slot),
                blockers=tuple(map(int,sorted(visited))))


class LeftGoalYield:
    """Round-local plan; keep the released goal until the carrier clears entry."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.key = self.blocked_key = None
        self.started = self.last_tick = None
        self.plan = None
        self.events = []

    def update(self, scenario, snapshot, goals, *, site, threats=(), coordinated_wait=False):
        key = (snapshot.round_number, site)
        if self.key != key:
            self.reset()
            self.key = key
        if site != 'L' or snapshot.phase == 'setup' or snapshot.is_planted:
            self.plan = None
            return dict(goals)
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        if self.plan is not None:
            alive = {a.slot: a for a in snapshot.allies if a.alive}
            if (holder is None or holder.slot != self.plan['carrier_slot']
                    or holder.position == self.plan['entry'] or holder.plant_progress
                    or self.plan['slot'] not in alive or goals.get(holder.slot) != self.plan['entry']):
                self.plan = None
                self.blocked_key = self.started = None
            else:
                result = dict(goals)
                result[self.plan['slot']] = self.plan['goal']
                return result
        proposal = left_yield_proposal(scenario, snapshot, goals, threats) if not coordinated_wait else None
        blocked_key = (proposal['carrier_slot'], proposal['entry']) if proposal else None
        if (blocked_key != self.blocked_key or self.last_tick is None
                or snapshot.tick not in (self.last_tick, self.last_tick+1)):
            self.started = snapshot.tick
        self.last_tick, self.blocked_key = snapshot.tick, blocked_key
        if proposal is not None and snapshot.tick-self.started >= BLOCKED_TICKS:
            self.plan = proposal
            self.events.append(dict(tick=snapshot.tick, old_goal=goals[proposal['slot']], **proposal))
            result = dict(goals)
            result[proposal['slot']] = proposal['goal']
            return result
        return dict(goals)
