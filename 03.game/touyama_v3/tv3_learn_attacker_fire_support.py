"""Plant-only public shot geometry, recent threats and defensive examples."""
from dataclasses import replace
import numpy as np
from grid_lines import line_cells
from game_core import FACING_VECTORS
from frc_v1.perception import Sighting
from frc_v1.actions import MOVE_STEPS, KINDS
from touyama_v3.tv3_attacker_combat import RECENT_CONTACT_TICKS, LOW_HP_FRACTION, AttackerCombatCoach

MOVES = ('STAY', 'N', 'E', 'S', 'W')
FIRE_SUPPORT_FEATURES = len(MOVES)*4


def fire_support_schema():
    return dict(version=1, features=FIRE_SUPPORT_FEATURES, moves=MOVES,
                channels=('potential_lines', 'minimum_target_support', 'unsupported_lines', 'stationary_contact'),
                memory_ticks=RECENT_CONTACT_TICKS, blockers='public_allies_and_current_sightings',
                smoke='whole_line_except_adjacent_or_public_recon_target',
                utility='recent_public_threats_and_smoke_on_fire_lines',
                teacher='single_unsupported_contact_cover_if_available', selection='learned_legal_actions')


def public_threats(snapshot, tracks):
    alive = {e.enemy_id for e in snapshot.enemies if e.alive}
    result = [s for s in snapshot.sightings if s.enemy_id in alive]
    current = {s.enemy_id for s in result}
    occupied = {s.position for s in result}
    visible = set(snapshot.visible_cells)
    for enemy_id, (position, tick, _) in tracks.items():
        if (enemy_id in alive and enemy_id not in current
                and 0 <= snapshot.tick-tick <= RECENT_CONTACT_TICKS
                and (position not in visible or position in occupied)):
            result.append(Sighting(enemy_id, position, 'history'))
    return tuple(result)


def shot_clear(scenario, snapshot, origin, target, slot):
    if origin == target or not scenario.clear(origin, target):
        return False
    line = line_cells(origin,target)
    if len(line) <= 2:
        return True  # Match the engine's adjacent shot rule.
    blockers = {a.position for a in snapshot.allies if a.alive and a.slot != slot}
    blockers.update(s.position for s in snapshot.sightings if s.source != 'history' and s.position != target)
    if set(line[1:-1]).intersection(blockers):
        return False
    revealed = any(s.position == target and s.source == 'reveal' for s in snapshot.sightings)
    return revealed or not set(line).intersection(snapshot.smoke_cells)


def target_support(scenario, snapshot, target, own_slot):
    count = 0
    for ally in snapshot.allies:
        if not ally.alive or ally.slot == own_slot or ally.blind:
            continue
        dx,dy = FACING_VECTORS[ally.facing]
        if ((target[1]-ally.position[1])*dx+(target[0]-ally.position[0])*dy >= 0
                and shot_clear(scenario,snapshot,ally.position,target,ally.slot)):
            count += 1
    return count


def contact_geometry(scenario, snapshot, ally, position, threats):
    virtual = replace(snapshot, allies=tuple(replace(a,position=position) if a.slot == ally.slot else a
                                            for a in snapshot.allies))
    contacts = [s for s in threats if shot_clear(scenario,virtual,position,s.position,ally.slot)]
    support = [target_support(scenario,virtual,s.position,ally.slot) for s in contacts]
    return contacts,support


def fire_support_features(scenario,snapshot,ally,tracks):
    threats = public_threats(snapshot,tracks)
    values = []
    occupied = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
    for move in MOVES:
        dr,dc = MOVE_STEPS.get(move,(0,0))
        position = ally.position[0]+dr,ally.position[1]+dc
        if not (0 <= position[0] < scenario.grid.shape[0] and 0 <= position[1] < scenario.grid.shape[1]) or scenario.grid[position] == 1 or position in occupied:
            values += [0.]*4
            continue
        contacts,support = contact_geometry(scenario,snapshot,ally,position,threats)
        values += [len(contacts)/5,min(support,default=0)/4,
                   sum(n == 0 for n in support)/5,float(move == 'STAY' and bool(contacts))]
    return np.asarray(values,np.float32)


def defensive_advice(scenario,snapshot,ally,advice,tracks,masks):
    threats = public_threats(snapshot,tracks)
    contacts,support = contact_geometry(scenario,snapshot,ally,ally.position,threats)
    target = min(contacts,key=lambda s:(max(abs(s.position[0]-ally.position[0]),abs(s.position[1]-ally.position[1])),s.enemy_id)) if contacts else None
    primary_support = support[contacts.index(target)] if target else 0
    from touyama_v3.tv3_observer import facing
    result = replace(advice, contacts=len(contacts), supporters=primary_support,
                     facing=ally.facing if ally.forced_facing or not target else facing(ally.position, target.position),
                     target=target.position if target else advice.target)
    if not contacts:
        return result
    # Count support for the actual target, not allies covering unrelated enemies.
    if primary_support == 0 or ally.blind or ally.hp <= ally.max_hp*LOW_HP_FRACTION:
        options = []
        for kind,(dr,dc) in MOVE_STEPS.items():
            if not masks.kind[ally.slot,KINDS.index(kind)]:
                continue
            position=ally.position[0]+dr,ally.position[1]+dc
            future,_=contact_geometry(scenario,snapshot,ally,position,threats)
            if len(future) < len(contacts):
                options.append((len(future),position))
        if options:
            result=replace(result,position=min(options)[1],reason='cover_retreat')
    elif len(contacts)>=2 and min(support)>=len(contacts) and not ally.blind and ally.hp>ally.max_hp*LOW_HP_FRACTION:
        result=replace(result,position=ally.position,reason='stop_shoot')
    return result


def smoke_screen(scenario,snapshot,ally,threats,target_mask):
    """Prefer a nearby screen on a public fire line over an enemy region marker."""
    if not threats:
        return None
    lines=[]
    for own in snapshot.allies:
        if own.alive and not own.reveal:
            # Smoke cannot protect a teammate publicly marked by recon. Do not
            # repeatedly screen a line which is already covered or body-blocked.
            lines.extend(line_cells(own.position,s.position) for s in threats
                         if shot_clear(scenario,snapshot,own.position,s.position,own.slot)
                         and not set(line_cells(own.position,s.position)).intersection(snapshot.smoke_cells))
    if not lines:
        return None
    occupied={a.position for a in snapshot.allies if a.alive}
    candidates={p for line in lines for p in line[1:-1] if p not in occupied
                and max(abs(p[0]-ally.position[0]),abs(p[1]-ally.position[1]))<=10
                and target_mask[p]}
    def rank(p):
        cells={(p[0]+dr,p[1]+dc) for dr in (-1,0,1) for dc in (-1,0,1)}
        coverage=sum(bool(cells.intersection(line[1:-1])) for line in lines if len(line)>2)
        return coverage,-max(abs(p[0]-ally.position[0]),abs(p[1]-ally.position[1])),p
    return max(candidates,key=rank) if candidates else None


class PublicPlantCombatCoach(AttackerCombatCoach):
    def clear(self, snapshot, origin, target, slot):
        return shot_clear(self.scenario, snapshot, origin, target, slot)
