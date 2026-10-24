"""Public carrier-support paths and teaching inputs; never select model actions."""
from collections import deque
from dataclasses import replace
import numpy as np
from game_core import FACING_VECTORS
from frc_v1.actions import MOVE_STEPS
from grid_paths import distance_map
from touyama_v3.tv3_observer import facing
from touyama_v3.tv3_attacker_combat import LOW_HP_FRACTION
from touyama_v3.tv3_learn_attacker_fire_support import public_threats, shot_clear, target_support

SUPPORT_SEARCH_STEPS = 4
SUPPORT_FEATURES = 17
MOVES = ('STAY', 'N', 'E', 'S', 'W')


def support_schema():
    return dict(version=1, features=SUPPORT_FEATURES, search_steps=SUPPORT_SEARCH_STEPS,
                threats='current_or_valid_recent_public_carrier_contacts_within_four_entry_steps',
                teacher='short_unoccupied_path_to_same_target_firing_lane',
                helpers='already_inside_selected_site_not_blind_or_low_hp',
                activation='reached_goal_without_firing_contact_and_carrier_without_ready_support',
                commitment='same_round_site_carrier_until_threat_invalid_or_plant_begins',
                lane_goals='do_not_occupy_carrier_plant_goal',
                movement='legacy_route_corridor_edge_window_and_team_reservations',
                selection='learned_actions_no_support_override')


def support_targets(scenario, snapshot, route, tracks):
    holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
    if holder is None or not 0 <= scenario.site_dist[route.site][holder.position] <= SUPPORT_SEARCH_STEPS:
        return ()
    positions = [holder.position]
    field = scenario.site_dist[route.site]
    positions += [p for p in scenario.neighbors(holder.position)
                  if 0 <= field[p] < field[holder.position]]
    return tuple(s for s in public_threats(snapshot, tracks)
                 if any(shot_clear(scenario, snapshot, p, s.position, holder.slot) for p in positions))


def support_path(scenario, snapshot, ally, targets, route=None):
    """Return a short reachable lane and first step, including body blockers."""
    if (not targets or ally.has_spike or ally.blind or ally.hp <= ally.max_hp * LOW_HP_FRACTION
            or scenario.grid[ally.position] != 2):
        return None
    occupied = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
    queue = deque([(ally.position, ally.position, 0)])
    seen = {ally.position}
    while queue:
        pos, first, depth = queue.popleft()
        virtual = replace(snapshot, allies=tuple(replace(a, position=pos) if a.slot == ally.slot else a
                                                for a in snapshot.allies))
        contacts = [s for s in targets if shot_clear(scenario, virtual, pos, s.position, ally.slot)]
        if contacts and (route is None or pos != route.cells[-1]):
            target = min(contacts, key=lambda s: (max(abs(pos[0]-s.position[0]), abs(pos[1]-s.position[1])), s.enemy_id))
            return dict(goal=pos, step=first, steps=depth, target=target.position)
        if depth < SUPPORT_SEARCH_STEPS:
            for p in sorted(scenario.neighbors(pos)):
                if p not in seen and p not in occupied:
                    seen.add(p)
                    queue.append((p, p if depth == 0 else first, depth+1))
    return None


def support_features(scenario, snapshot, ally, targets, plan):
    values = []
    field = distance_map(scenario.grid, plan['goal']) if plan else None
    occupied = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
    for move in MOVES:
        dr, dc = MOVE_STEPS.get(move, (0, 0))
        pos = ally.position[0]+dr, ally.position[1]+dc
        valid = (0 <= pos[0] < scenario.grid.shape[0] and 0 <= pos[1] < scenario.grid.shape[1]
                 and scenario.grid[pos] != 1 and pos not in occupied)
        if not valid:
            values += [0., 0., 0.]
            continue
        virtual = replace(snapshot, allies=tuple(replace(a, position=pos) if a.slot == ally.slot else a
                                                for a in snapshot.allies))
        lines = [s for s in targets if shot_clear(scenario, virtual, pos, s.position, ally.slot)]
        dx, dy = FACING_VECTORS[ally.facing]
        ready = sum((s.position[1]-pos[1])*dx+(s.position[0]-pos[0])*dy >= 0 for s in lines)
        progress = np.clip(field[ally.position]-field[pos], -1, 1) if field is not None and field[pos] >= 0 else 0.
        values += [len(lines)/5, ready/5, float(progress)]
    values += [len(targets)/5, float(plan is not None)]
    return np.asarray(values, np.float32)


def support_advice(scenario, snapshot, ally, advice, targets, plan):
    if plan is None:
        return advice
    holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
    side = scenario.site_of(plan['goal'])
    virtual = replace(snapshot, allies=tuple(replace(a, position=plan['step']) if a.slot == ally.slot else a
                                            for a in snapshot.allies))
    exposed = {s.position for s in (*targets, *snapshot.sightings)
               if shot_clear(scenario, virtual, plan['step'], s.position, ally.slot)}
    entry_cover = (side is not None and advice.reason == 'cover_retreat' and advice.contacts == 1 and len(exposed) <= 1
                   and holder is not None and not holder.blind and holder.hp > holder.max_hp * LOW_HP_FRACTION
                   and 0 <= scenario.site_dist[side][holder.position] <= 2)
    if advice.reason not in ('advance', 'assemble', 'stop_shoot', 'crossfire_reposition') and not entry_cover:
        return advice
    return replace(advice, position=plan['step'],
                   facing=ally.facing if ally.forced_facing else facing(plan['step'], plan['target']),
                   target=plan['target'], reason='crossfire_reposition' if plan['steps'] else 'stop_shoot')


class SupportCoach:
    """Keep a teaching path after leaving a reached, ineffective endpoint."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.context = None
        self.active = {}

    def plan(self, scenario, snapshot, ally, goal, route, targets, advice):
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        context = snapshot.round_number, route.site, holder.slot if holder else None
        if context != self.context:
            self.reset()
            self.context = context
        if holder is None or holder.plant_progress:
            self.active.clear()
            return None
        active = self.active.get(ally.slot)
        selected = tuple(s for s in targets if s.enemy_id == active) if active is not None else ()
        if not selected:
            self.active.pop(ally.slot, None)
            if ally.position != goal or advice.reason != 'advance':
                return None
            selected = tuple(s for s in targets if target_support(scenario, snapshot, s.position, holder.slot) == 0)
        plan = support_path(scenario, snapshot, ally, selected, route)
        if plan:
            self.active[ally.slot] = next(s.enemy_id for s in selected if s.position == plan['target'])
        else:
            self.active.pop(ally.slot, None)
        return plan
