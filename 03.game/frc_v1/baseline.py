"""Explicit rule teacher for bootstrapping; this is not a trained policy."""

from collections import deque
import math
import numpy as np

from frc_v1 import FACING, FACING_STEPS
from frc_v1.actions import FrcAction, TeamDecision, KINDS, MOVE_STEPS


def plant_sites(grid):
    cells = {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 2}
    sites = []
    while cells:
        start = min(cells)
        cells.remove(start)
        group, pending = [start], [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in MOVE_STEPS.values():
                neighbor = r + dr, c + dc
                if neighbor in cells:
                    cells.remove(neighbor)
                    pending.append(neighbor)
                    group.append(neighbor)
        sites.append(tuple(sorted(group)))
    return tuple(sorted(sites))


def facing_to(origin, target):
    dr, dc = target[0] - origin[0], target[1] - origin[1]
    if dr == dc == 0:
        return "N"
    return FACING[max(range(8), key=lambda i: (FACING_STEPS[i][0] * dr + FACING_STEPS[i][1] * dc) /
                      math.hypot(*FACING_STEPS[i]))]


def route_step(grid, start, goals, blocked=(), first_step_blocked=()):
    goals = set(goals)
    if start in goals:
        return "STAY", 0
    blocked = set(blocked) - goals
    first_step_blocked = set(first_step_blocked)
    queue = deque([(start, "STAY", 0)])
    visited = {start}
    while queue:
        (r, c), first, distance = queue.popleft()
        for kind, (dr, dc) in MOVE_STEPS.items():
            pos = r + dr, c + dc
            if (not (0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0]))
                    or grid[pos[0]][pos[1]] == 1 or pos in blocked or pos in visited
                    or distance == 0 and pos in first_step_blocked):
                continue
            move = kind if distance == 0 else first
            if pos in goals:
                return move, distance + 1
            visited.add(pos)
            queue.append((pos, move, distance + 1))
    return "STAY", 10000


class FrcBaseline:
    def reset(self):
        pass

    def act(self, observation, snapshot, belief):
        masks = observation.masks
        sites = plant_sites(snapshot.grid)
        if not sites:
            sites = ((snapshot.allies[2].position,),)
        site = min(range(len(sites)), key=lambda i: route_step(snapshot.grid, snapshot.allies[2].position, sites[i])[1])
        goals = (snapshot.spike_planted,) if snapshot.is_planted and snapshot.spike_planted else sites[site]
        entry = 2 if snapshot.allies[2].alive else next((a.slot for a in (snapshot.allies[4], *snapshot.allies) if a.alive), 2)
        follow = 4 if snapshot.allies[4].alive and entry != 4 else next(
            (a.slot for a in snapshot.allies if a.alive and a.slot != entry), entry)
        carrier = next((a.slot for a in snapshot.allies if a.alive and a.has_spike), None)
        objective = carrier if carrier is not None else next((a.slot for a in snapshot.allies if a.alive), 0)
        if snapshot.side == "D" and snapshot.is_planted:
            objective = min((a.slot for a in snapshot.allies if a.alive), default=0,
                key=lambda i: route_step(snapshot.grid, snapshot.allies[i].position, goals)[1])
        phase = "POSTPLANT" if snapshot.side == "A" and snapshot.is_planted else (
            "RETAKE" if snapshot.is_planted else "DEFEND" if snapshot.side == "D" else
            "PLANT" if masks.kind[objective, 5] else "ENTRY")
        histories = {h.handle: h for h in belief.effects}
        danger = set()
        for effect in snapshot.effects:
            if effect.kind in ("NEON", "TUNNEL", "DESTRUCTION", "BALEMOON") and histories[effect.handle].affiliation != "own":
                danger.update(effect.cells)
        actions = []
        reserved = {a.position for a in snapshot.allies if a.alive}
        for a in snapshot.allies:
            facing = a.facing
            enemies = sorted(snapshot.sightings, key=lambda s: math.dist(a.position, s.position))
            if enemies and not a.forced_facing:
                facing = facing_to(a.position, enemies[0].position)
            elif not a.forced_facing:
                facing = facing_to(a.position, goals[0])
            action = FrcAction(facing=facing)
            if not a.alive or not masks.kind[a.slot, 1:].any():
                actions.append(FrcAction(facing=a.facing))
                continue
            if a.position in danger:
                safe = [(r, c) for r, row in enumerate(snapshot.grid) for c, value in enumerate(row)
                        if value != 1 and (r, c) not in danger and
                        (snapshot.phase != "setup" or (r, c) in snapshot.setup_cells)]
                kind, _ = route_step(snapshot.grid, a.position, safe, first_step_blocked=reserved - {a.position})
                action = FrcAction(kind, facing)
            elif a.slot == 0 and masks.kind[0, 8]:
                targets = [ally for ally in snapshot.allies if masks.target[0, 0, ally.slot]]
                target = min(targets, key=lambda ally: (ally.hp > 50, ally.slot != entry, ally.hp))
                if target.hp <= 70:
                    action = FrcAction("ABILITY", facing, ally_slot=target.slot)
            elif masks.kind[a.slot, 5]:
                action = FrcAction("PLANT", facing)
            elif masks.kind[a.slot, 6] and a.slot == objective:
                action = FrcAction("DEFUSE", facing)
            elif a.slot == 4 and masks.kind[4, 9] and enemies and (not snapshot.allies[0].alive or
                    snapshot.is_planted and snapshot.allies[0].charges == 0 and a.hp <= 30):
                action = FrcAction("ULTIMATE", facing)
            elif a.slot == 3 and masks.kind[3, 9] and snapshot.tick >= 15:
                action = FrcAction("ULTIMATE", facing)
            elif a.slot in (1, 3, 4) and masks.kind[a.slot, 8] and (enemies or route_step(snapshot.grid, a.position, goals)[1] < 9):
                candidate = enemies[0].position if enemies else goals[0]
                candidates = np.flatnonzero(masks.target[a.slot, 0])
                columns = len(snapshot.grid[0])
                target_index = min(candidates, key=lambda i: math.dist(divmod(int(i), columns), candidate))
                # Keep the teacher from throwing all charges on consecutive ticks.
                own_flight = any(e.kind == ("SMOKE", "RECON", "ASH")[(1, 3, 4).index(a.slot)] and
                                 histories[e.handle].affiliation == "own" for e in snapshot.effects)
                if not own_flight and snapshot.tick % 8 == a.slot:
                    action = FrcAction("ABILITY", facing, divmod(int(target_index), columns))
            if action.kind == "STAY":
                own_goals = goals
                if snapshot.side == "A" and snapshot.spike_dropped and not snapshot.is_planted:
                    own_goals = (snapshot.spike_dropped,)
                elif snapshot.side == "A" and not snapshot.is_planted and a.slot not in (entry, follow):
                    lead = snapshot.allies[entry]
                    # The teacher asks support to trail the entry rather than race it to the site.
                    if math.dist(a.position, lead.position) < 3 and route_step(snapshot.grid, lead.position, goals)[1] > 0:
                        own_goals = (a.position,)
                elif snapshot.is_planted and snapshot.side == "A":
                    own_goals = tuple((r, c) for r, row in enumerate(snapshot.grid) for c, value in enumerate(row)
                                     if value != 1 and 2 <= max(abs(r - goals[0][0]), abs(c - goals[0][1])) <= 4)
                kind, _ = route_step(snapshot.grid, a.position, own_goals, danger,
                                     first_step_blocked=reserved - {a.position})
                if kind != "STAY" and masks.kind[a.slot, KINDS.index(kind)]:
                    action = FrcAction(kind, facing)
            if not masks.kind[a.slot, KINDS.index(action.kind)]:
                action = FrcAction(facing=a.facing)
            if action.kind in MOVE_STEPS:
                dr, dc = MOVE_STEPS[action.kind]
                reserved.add((a.position[0] + dr, a.position[1] + dc))
            actions.append(action)
        intents = tuple(2 if i == entry else 3 if i == follow else 0 if snapshot.is_planted else 1 for i in range(5))
        return TeamDecision(tuple(actions), site, phase, entry, follow, objective, intents)
