"""Public incoming fire and wall-cover teaching paths while marked by recon."""
from dataclasses import replace
from collections import deque
import numpy as np
from frc_v1.actions import KINDS, MOVE_STEPS
from grid_lines import line_cells
from game_core import REVEAL_DURATION_TICKS
from touyama_v3.tv3_learn_attacker_fire_support import public_threats, shot_clear

RECON_COVER_STEPS = 12
RECON_MOVES = ('STAY', 'N', 'E', 'S', 'W')
RECON_COVER_FEATURES = len(RECON_MOVES)*3 + 3


def recon_cover_schema():
    return dict(version=1, features=RECON_COVER_FEATURES, moves=RECON_MOVES,
        channels=('incoming_lines', 'unreturnable_lines', 'cover_first_step'),
        extra=('reveal_remaining', 'cover_available', 'cover_steps'),
        max_steps=RECON_COVER_STEPS, smoke='own_reveal_ignores_smoke_incoming_only',
        evidence='current_public_sightings_and_recent_valid_tracks',
        teacher='nonincreasing_incoming_lines_to_wall_cover',
        site_entry='stay_outside_sites_when_starting_outside',selection='learned_legal_actions')


def incoming_clear(scenario, snapshot, ally, position, threat):
    """Potential incoming line; unknown enemy facing/blindness is not invented."""
    if position == threat.position or not scenario.clear(threat.position, position):
        return False
    line = line_cells(threat.position, position)
    if len(line) <= 2:
        return True
    bodies = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot
              and (not getattr(snapshot,'temporary_body_cover',False)
                   or a.movement_disabled>0 and not a.warping)}
    bodies.update(s.position for s in snapshot.sightings
                  if s.source != 'history' and s.position != threat.position)
    if set(line[1:-1]).intersection(bodies):
        return False
    return ally.reveal > 0 or not set(line).intersection(snapshot.smoke_cells)


def recon_cover_plan(scenario, snapshot, ally, tracks, masks, destination=None):
    if ally.reveal <= 0:
        # Possible ally movement is an uncertainty input, not evidence that
        # the enemy currently has a shot. Do not teach an indefinite hold.
        return None
    threats = public_threats(snapshot, tracks)
    if not threats:
        return None
    blocked = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
    blocked.update(s.position for s in snapshot.sightings if s.source != 'history')
    cache = {}
    def risk(p):
        if p not in cache:
            cache[p] = sum(incoming_clear(scenario,snapshot,ally,p,t) for t in threats)
        return cache[p]
    origin = ally.position
    initial = risk(origin)
    if initial == 0:
        # Remaining behind a wall is useful; leaving it while marked is not.
        if destination is None or risk(destination)==0:
            return None
        plan=dict(step=origin,goal=origin,steps=0,incoming=0)
        if getattr(snapshot,'temporary_body_cover',False):
            plan['future_incoming']=risk(destination)
        return plan
    queue = deque([(origin,origin,0)])
    visited = {origin}
    candidates = []
    while queue:
        p, first, depth = queue.popleft()
        if depth == RECON_COVER_STEPS:
            continue
        for kind in RECON_MOVES[1:]:
            dr,dc = MOVE_STEPS[kind]
            q = p[0]+dr,p[1]+dc
            if not (0<=q[0]<scenario.grid.shape[0] and 0<=q[1]<scenario.grid.shape[1]):
                continue
            if q in visited or q in blocked or scenario.grid[q]==1:
                continue
            if scenario.grid[origin]!=2 and scenario.grid[q]==2:
                continue  # A retreat from an approach must not enter a defended site.
            if depth==0 and not masks.kind[ally.slot,KINDS.index(kind)]:
                continue
            if risk(q)>risk(p):
                continue
            visited.add(q)
            step = q if depth==0 else first
            if risk(q)<initial:
                candidates.append((risk(q),depth+1,q,step))
            queue.append((q,step,depth+1))
    if not candidates:
        return None
    incoming,steps,goal,step = min(candidates)
    return dict(step=step,goal=goal,steps=steps,incoming=incoming)


def recon_cover_advice(ally, advice, plan):
    if plan is None:
        return advice
    # No smoke-based outgoing-contact count is substituted for incoming risk.
    return replace(advice, position=plan['step'], reason='cover_retreat',
                   future_contacts=max(advice.future_contacts,plan.get('future_incoming',0)))


def recon_cover_features(scenario, snapshot, ally, tracks, plan):
    threats = public_threats(snapshot, tracks)
    blocked = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
    values = []
    for kind in RECON_MOVES:
        dr,dc = MOVE_STEPS.get(kind,(0,0))
        p = ally.position[0]+dr,ally.position[1]+dc
        if not (0<=p[0]<scenario.grid.shape[0] and 0<=p[1]<scenario.grid.shape[1]) or scenario.grid[p]==1 or p in blocked:
            values += [0.,0.,0.]
            continue
        incoming = [t for t in threats if incoming_clear(scenario,snapshot,ally,p,t)]
        values += [len(incoming)/5,
            sum(not shot_clear(scenario,snapshot,p,t.position,ally.slot,
                potential_body_cover=getattr(snapshot,'temporary_body_cover',False)) for t in incoming)/5,
            float(plan is not None and p==plan['step'])]
    values += [min(1.,ally.reveal/REVEAL_DURATION_TICKS),float(plan is not None),
               plan['steps']/RECON_COVER_STEPS if plan is not None else 1.]
    return np.asarray(values,np.float32)
