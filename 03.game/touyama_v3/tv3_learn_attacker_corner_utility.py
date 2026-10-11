"""Public FLASH impact proposals for learned preplant actions, including corner angles."""
import numpy as np
from frc_v1.actions import FrcAction,validate_action
from grid_lines import line_cells
from touyama_v3.tv3_attacker_entry_utility import projectile_impact
from touyama_v3.tv3_learn_attacker_fire_support import public_threats
from game_core import FLASH_MAX_FLIGHT_TICKS,BLIND_DURATION_TICKS
FLASH_COMPLETION_MARGIN_TICKS=1  # Expiry is exclusive; finish before its tick.

CORNER_UTILITY_FEATURES=8*5
CORNER_COORDINATE_SLOTS=(2,3,4)
FRIENDLY_FLASH_FEATURES=2


class FriendlyFlashTracker:
    def __init__(self,scenario):
        self.scenario=scenario
        self.records=[]

    def record(self,tick,origin,aim):
        key=(tick,tuple(origin),tuple(aim))
        if any(r.get('key')==key for r in self.records):
            return
        impact,delay=projectile_impact(self.scenario,origin,aim,'FLASH')
        self.records.append(dict(key=key,impact=impact,arrival=tick+delay,expiry=tick+delay+BLIND_DURATION_TICKS,confirmed=False))

    def suppressed(self,snapshot,position,tracks):
        fresh=any(s.position==position for s in snapshot.sightings) or any(
            p==position and 0<=snapshot.tick-t<=5 for p,t,_ in tracks.values())
        if not fresh:
            return False
        for r in self.records:
            if not r['arrival']<=snapshot.tick<r['expiry']:
                continue
            if not r['confirmed'] and any(e.kind=='FLASH' and e.phase=='active' and e.position==r['impact'] for e in snapshot.effects):
                r['confirmed']=True  # Public burst confirms the owned cast's forecast, not private enemy status.
                r['positions']={s.position for s in public_threats(snapshot,tracks)
                    if self.scenario.clear(r['impact'],s.position) and not set(
                        line_cells(r['impact'],s.position)).intersection(snapshot.smoke_cells)}
            if r['confirmed'] and position in r['positions']:
                return True
        return False

    def features(self,snapshot,tracks):
        threats=public_threats(snapshot,tracks)
        suppressed=sum(self.suppressed(snapshot,s.position,tracks) for s in threats)
        remaining=max((r['expiry']-snapshot.tick for r in self.records
                       if r['confirmed'] and r['positions'] and r['arrival']<=snapshot.tick<r['expiry']),default=0)
        return [suppressed/5,remaining/BLIND_DURATION_TICKS]


def confirmed_flash_covers_step(scenario,snapshot,ally,destination,tracks,tracker):
    """Known incoming lines covered by a confirmed owned burst, not a guarantee."""
    from touyama_v3.tv3_learn_attacker_recon_cover import incoming_clear
    if ally.blind:
        return False
    incoming=[s for s in public_threats(snapshot,tracks)
              if incoming_clear(scenario,snapshot,ally,destination,s)]
    return bool(incoming) and all(tracker.suppressed(snapshot,s.position,tracks) for s in incoming)


def confirmed_flash_step_window(scenario,snapshot,ally,destination,tracks,tracker):
    """Conservative remaining owned window across all known incoming lines."""
    from touyama_v3.tv3_learn_attacker_recon_cover import incoming_clear
    if not confirmed_flash_covers_step(scenario,snapshot,ally,destination,tracks,tracker):
        return 0
    incoming=[s for s in public_threats(snapshot,tracks)
              if incoming_clear(scenario,snapshot,ally,destination,s)]
    return min(max((r['expiry']-snapshot.tick for r in tracker.records
                    if r['confirmed'] and s.position in r.get('positions',())
                    and r['arrival']<=snapshot.tick<r['expiry']),default=0) for s in incoming)


def flash_window_allows_completion(window,travel,required_ticks):
    return travel>=0 and window>=travel+max(1,required_ticks)+FLASH_COMPLETION_MARGIN_TICKS


def corner_utility_schema():
    return dict(version=1,features=CORNER_UTILITY_FEATURES,ability='FLASH',
        fields=('impact_row','impact_column','flight_ticks','known_covered','current_covered'),
        proposal_slots=CORNER_COORDINATE_SLOTS,targets='selected_site_floor_and_public_threats',
        selection='learned_legal_actions_no_forced_cast')


def covered_threats(scenario,snapshot,origin,aim,threats):
    impact,delay=projectile_impact(scenario,origin,aim,'FLASH')
    if not 0<delay<=FLASH_MAX_FLIGHT_TICKS:
        return impact,delay,()
    smoke=set(snapshot.smoke_cells)
    hits=tuple(s for s in threats if scenario.clear(impact,s.position)
               and not smoke.intersection(line_cells(impact,s.position)))
    return impact,delay,hits


def fill_corner_targets(scenario,snapshot,ally,route,tracks,masks,direction,coordinates):
    result=list(coordinates)
    if ally.ability_name!='FLASH':
        return result
    threats=public_threats(snapshot,tracks)
    if not threats:
        return result
    options=[]
    for aim in sorted(set(scenario.sites[route.site])|{s.position for s in threats}):
        aim=tuple(map(int,aim))
        try:
            validate_action(snapshot,masks,ally.slot,FrcAction('ABILITY',direction,aim))
        except (ValueError,TypeError,IndexError):
            continue
        impact,delay,hits=covered_threats(scenario,snapshot,ally.position,aim,threats)
        if hits:
            options.append((-len(hits),delay,max(abs(aim[0]-ally.position[0]),abs(aim[1]-ally.position[1])),aim,impact))
    options.sort()
    used=set()
    for aim in result:
        if aim is not None:
            _,_,hits=covered_threats(scenario,snapshot,ally.position,aim,threats)
            if hits:
                used.add(aim)
    for slot in CORNER_COORDINATE_SLOTS:
        if result[slot] is not None and covered_threats(scenario,snapshot,ally.position,result[slot],threats)[2]:
            continue
        choice=next((row for row in options if row[3] not in used),None)
        if choice is not None:
            result[slot]=choice[3]
            used.add(choice[3])
    return result


def corner_utility_features(scenario,snapshot,ally,tracks,actions,mask):
    values=[]
    threats=public_threats(snapshot,tracks)
    current={s.enemy_id for s in snapshot.sightings}
    for i in range(40,48):
        aim=actions[i].target
        if ally.ability_name!='FLASH' or aim is None or not mask[i]:
            values.extend([0.]*5)
            continue
        impact,delay,hits=covered_threats(scenario,snapshot,ally.position,aim,threats)
        values.extend((impact[0]/max(1,scenario.grid.shape[0]-1),
                       impact[1]/max(1,scenario.grid.shape[1]-1),delay/FLASH_MAX_FLIGHT_TICKS,
                       len(hits)/5,sum(s.enemy_id in current for s in hits)/5))
    return np.asarray(values,np.float32)
