"""Experimental public-only teacher, installed only by the approach diagnostic.

This is not connected to training or production inference. Validate its effects
before adopting observation/action changes into a new checkpoint contract.
"""
from dataclasses import replace
import heapq
from collections import deque
import numpy as np
from grid_lines import line_cells
from frc_v1.actions import MOVE_STEPS, FrcAction, build_masks, validate_action
from frc_v1.perception import Sighting
from touyama_v3.tv3_observer import facing
from touyama_v3.tv3_attacker_combat import LOW_HP_FRACTION
from touyama_v3.tv3_learn_attacker_fire_support import shot_clear, target_support
from touyama_v3.tv3_attacker_combat import effective_utility

RISK_MEMORY_TICKS = 40
ROUTE_STALL_TICKS = 18
UNSUPPORTED_EXPOSURE_COST = 32
ROUTE_DANGER_LOOKAHEAD = 8
MAX_DETOUR_EXTRA_STEPS = 8
EXPERIMENT_VERSION = 'asymmetric_smoke_deadline_v7'
KEEP_FEASIBLE_PLANT_PLAN = True
ONLY_ASYMMETRIC_SMOKE = True
WAIT_FOR_UNSUPPORTED_LINE = False  # Ablation: perpetual public-risk waiting caused full-round timeouts.
ENABLE_EXPOSURE_DETOURS = False  # Ablation: isolate directional incoming LOS and utility from route changes.
ENABLE_PROGRESS_OVERRIDE = False


def projected_recon_area(scenario,snapshot,position):
    from game_core import RECON_REVEAL_SIZE
    for effect in snapshot.effects:
        if effect.kind != 'RECON':
            continue
        if effect.phase == 'active' and position in effect.cells:
            return True
        if effect.phase != 'flight' or effect.position is None or effect.direction == (0.,0.):
            continue
        dr,dc=effect.direction
        extent=max(scenario.grid.shape)*2
        aim=(effect.position[0]+round(dr*extent),effect.position[1]+round(dc*extent))
        path=scenario._projectile_path(effect.position,aim)
        if path and max(abs(position[0]-path[-1][0]),abs(position[1]-path[-1][1])) <= RECON_REVEAL_SIZE//2:
            return True
    return False  # Affiliation is unknown: this predicts possible marking, not a guaranteed enemy cast.


def safe_plant_detour(scenario,snapshot,ally,site,threats):
    """Public weighted path, used only for experimental teacher examples."""
    blocked={a.position for a in snapshot.allies if a.alive and a.slot!=ally.slot}
    blocked |= {s.position for s in threats}
    goals=set(scenario.sites[site])-blocked
    other_site=set(scenario.sites['L' if site=='R' else 'R'])
    blocked |= other_site
    from game_core import PLANT_REQUIRED_TICKS
    from touyama_v3.tv3_learn_attacker_analysis import TIME_MARGIN_TICKS
    # Share the analysis deadline, and do not spend the entire round buying safety.
    distances={p:0 for p in goals}
    queue=deque(goals)
    while queue:
        p=queue.popleft()
        for q in scenario.neighbors(p):
            if q not in blocked and q not in distances:
                distances[q]=distances[p]+1
                queue.append(q)
    shortest=distances.get(ally.position)
    if shortest is None:
        return None
    budget=min(snapshot.round_timer-PLANT_REQUIRED_TICKS-TIME_MARGIN_TICKS,
               shortest+MAX_DETOUR_EXTRA_STEPS)
    if shortest>budget:
        return None
    heap=[(0,0,ally.position)]
    costs={(ally.position,0):0}
    parents={(ally.position,0):None}
    labels={ally.position:[(0,0)]}
    exposure_cache={}
    while heap:
        cost,steps,p=heapq.heappop(heap)
        state=(p,steps)
        if cost!=costs.get(state) or any(s<=steps and c<=cost and (s,c)!=(steps,cost)
                                       for s,c in labels.get(p,())):
            continue
        if p in goals:
            path=[]
            while state is not None:
                path.append(state[0])
                state=parents[state]
            path.reverse()
            return dict(path=tuple(path),steps=steps,cost=cost,budget=budget,shortest=shortest)
        for q in scenario.neighbors(p):
            if q in blocked or steps+1+distances.get(q,budget+1)>budget:
                continue
            if q not in exposure_cache:
                exposure_cache[q]=sum(incoming_clear(scenario,snapshot,ally,q,s) for s in threats)
            exposure=exposure_cache[q]
            new=cost+1+UNSUPPORTED_EXPOSURE_COST*exposure
            if any(s<=steps+1 and c<=new for s,c in labels.get(q,())):
                continue
            labels[q]=[(s,c) for s,c in labels.get(q,()) if not (s>=steps+1 and c>=new)] + [(steps+1,new)]
            costs[(q,steps+1)]=new
            parents[(q,steps+1)]=(p,steps)
            heapq.heappush(heap,(new,steps+1,q))
    return None


def risk_threats(snapshot, tracks):
    current = {s.enemy_id:s for s in snapshot.sightings}
    alive = {e.enemy_id for e in snapshot.enemies if e.alive}
    visible = set(snapshot.visible_cells)
    result = [s for s in current.values() if s.enemy_id in alive]
    for enemy_id,(position,tick,_) in tracks.items():
        if (enemy_id in alive and enemy_id not in current and 0 <= snapshot.tick-tick <= RISK_MEMORY_TICKS
                and position not in visible):
            result.append(Sighting(enemy_id,position,'history'))
    return tuple(result)


def incoming_clear(scenario, snapshot, ally, position, threat):
    if position == threat.position or not scenario.clear(position,threat.position):
        return False
    line = line_cells(position,threat.position)
    if len(line) <= 2:
        return True
    blockers = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
    blockers |= {s.position for s in snapshot.sightings if s.enemy_id != threat.enemy_id}
    if blockers.intersection(line[1:-1]):
        return False
    # The target of the incoming shot is our ally, not the observed enemy.
    return bool(ally.reveal or projected_recon_area(scenario,snapshot,position)) or not set(line).intersection(snapshot.smoke_cells)


def exposure_advice(scenario, snapshot, ally, inputs, tracks, site=None, route=None):
    threats = risk_threats(snapshot,tracks)
    current = [s for s in threats if incoming_clear(scenario,snapshot,ally,ally.position,s)]
    action = inputs.actions[inputs.teacher]
    dr,dc = MOVE_STEPS.get(action.kind,(0,0))
    destination = ally.position[0]+dr,ally.position[1]+dc
    future = [s for s in threats if incoming_clear(scenario,snapshot,ally,destination,s)]
    target = (current or future)[0] if current or future else None
    supporter = target_support(scenario,snapshot,target.position,ally.slot) if target else 0
    outgoing = any(shot_clear(scenario,snapshot,ally.position,s.position,ally.slot) for s in current)
    advice = inputs.combat
    upcoming=[]
    if route is not None and ally.position in route.cells:
        index=route.cells.index(ally.position)
        for p in route.cells[index+1:index+1+ROUTE_DANGER_LOOKAHEAD]:
            virtual=replace(snapshot,allies=tuple(replace(a,position=p) if a.slot==ally.slot else a for a in snapshot.allies))
            for s in threats:
                if incoming_clear(scenario,snapshot,ally,p,s) and target_support(scenario,virtual,s.position,ally.slot)==0:
                    upcoming.append((p,s.position))
    asymmetric_current=any(not shot_clear(scenario,snapshot,ally.position,s.position,ally.slot) for s in current)
    asymmetric_future=any(not shot_clear(scenario,snapshot,destination,s.position,ally.slot) for s in future)
    intervention_allowed=not ONLY_ASYMMETRIC_SMOKE or asymmetric_current or asymmetric_future
    if intervention_allowed and current and (not outgoing or ally.hp <= ally.max_hp*LOW_HP_FRACTION or ally.blind):
        options = []
        for i in np.flatnonzero(inputs.mask[:40]):
            a=inputs.actions[i]
            dr,dc=MOVE_STEPS.get(a.kind,(0,0))
            p=ally.position[0]+dr,ally.position[1]+dc
            count=sum(incoming_clear(scenario,snapshot,ally,p,s) for s in threats)
            if count < len(current):
                field=getattr(inputs,'distances',None)
                remaining=field[p] if field is not None and field[p]>=0 else 9999
                options.append((count,remaining,p))
        if options:
            advice=replace(advice,position=min(options)[2],reason='cover_retreat',
                           target=target.position,contacts=len(current),supporters=supporter)
    elif future and not current and supporter == 0 and action.kind in MOVE_STEPS:
        # Elapsed waiting does not clear an unchanged enemy line.
        if WAIT_FOR_UNSUPPORTED_LINE:
            advice=replace(advice,position=ally.position,reason='assemble',target=target.position,
                           contacts=0,future_contacts=len(future),supporters=0)
        if ENABLE_EXPOSURE_DETOURS and site is not None and scenario.grid[ally.position] != 2:
            detour=safe_plant_detour(scenario,snapshot,ally,site,threats)
            if detour and len(detour['path'])>1:
                step=detour['path'][1]
                if not any(incoming_clear(scenario,snapshot,ally,step,s) for s in threats):
                    advice=replace(advice,position=step,reason='crossfire_reposition')
                    inputs.diagnostic_allow_detour=True
                    inputs.exposure_detour_plan=detour
    if ENABLE_EXPOSURE_DETOURS and upcoming and not current and site is not None and scenario.grid[ally.position] != 2:
        detour=safe_plant_detour(scenario,snapshot,ally,site,threats)
        if detour and len(detour['path'])>1:
            step=detour['path'][1]
            if not any(incoming_clear(scenario,snapshot,ally,step,s) for s in threats):
                advice=replace(advice,position=step,reason='crossfire_reposition')
                inputs.diagnostic_allow_detour=True
                inputs.exposure_detour_plan=detour
    return advice,dict(current=len(current),future=len(future),supporters=supporter,
                       intervention_allowed=intervention_allowed,
                       outgoing=outgoing,own_recon_reveal=ally.reveal,
                       upcoming_route_lines=upcoming,
                       threats=[(s.enemy_id,s.position,s.source) for s in threats])


class ExposureEncoderProbe:
    def __init__(self, base):
        self.base = base
        self.detours = {}

    def __getattr__(self, name):
        return getattr(self.base,name)

    def encode(self,snapshot,ally,goal,route,analysis,tracks,masks,cursor,mode,coordination=None):
        inputs=self.base.encode(snapshot,ally,goal,route,analysis,tracks,masks,cursor,mode,coordination)
        advice,debug=exposure_advice(self.base.scenario,snapshot,ally,inputs,tracks,route.site,route)
        threats=risk_threats(snapshot,tracks)
        active=self.detours.get(ally.slot)
        if getattr(inputs,'diagnostic_allow_detour',False):
            self.detours[ally.slot]=(route.site,{s.enemy_id for s in threats})
        elif active and ENABLE_EXPOSURE_DETOURS:
            if (active[0] != route.site or not active[1].intersection(s.enemy_id for s in threats)
                    or self.base.scenario.grid[ally.position] == 2):
                self.detours.pop(ally.slot,None)
            elif not debug['current'] and not ally.blind:
                detour=safe_plant_detour(self.base.scenario,snapshot,ally,route.site,threats)
                if detour and len(detour['path'])>1:
                    advice=replace(advice,position=detour['path'][1],reason='crossfire_reposition')
                    inputs.diagnostic_allow_detour=True
                    inputs.exposure_detour_plan=detour
        debug['detour']=getattr(inputs,'exposure_detour_plan',None)
        inputs.exposure_debug=debug
        if advice != inputs.combat:
            direction=ally.facing if ally.forced_facing else facing(ally.position,advice.target or goal)
            for i in np.flatnonzero(inputs.mask[:40]):
                action=inputs.actions[i]
                dr,dc=MOVE_STEPS.get(action.kind,(0,0))
                if (ally.position[0]+dr,ally.position[1]+dc)==advice.position and action.facing==direction:
                    inputs.teacher=int(i)
                    break
            inputs.combat=advice
        # Counter an asymmetric recon/smoke line with public-source utility.
        if debug.get('intervention_allowed',True) and (debug['current'] or debug['future']) and advice.reason != 'cover_retreat':
            public=replace(snapshot,sightings=threats)
            if ally.ability_name in ('FLASH','RECON') and not any(
                    e.kind==ally.ability_name and e.phase in ('active','flight') for e in snapshot.effects):
                for threat in threats:
                    proposal=FrcAction('ABILITY',ally.facing if ally.forced_facing else facing(ally.position,threat.position),threat.position)
                    try:
                        validate_action(snapshot,masks,ally.slot,proposal)
                    except (ValueError,TypeError,IndexError):
                        continue
                    if effective_utility(self.base.scenario,public,ally,{'ability':ally.ability_name,'target':threat.position}):
                        actions=list(inputs.actions)
                        actions[40]=proposal
                        inputs.actions=tuple(actions)
                        inputs.mask[40]=True
                        inputs.teacher=40
                        break
        return inputs


class PlantDeadlineProbe:
    """Keep learned legal actions available after an off-route arrival inside the chosen site."""
    def __init__(self,base):
        self.base=base

    def __getattr__(self,name):
        return getattr(self.base,name)

    @property
    def progress_tick(self):
        return self.base.progress_tick

    @progress_tick.setter
    def progress_tick(self,value):
        self.base.progress_tick=value

    def update(self,snapshot,observation,encoder,model,route=None,mode=None,**kwargs):
        plan=self.base.update(snapshot,observation,encoder,model,route,mode,**kwargs)
        if plan is not None or route is None:
            return plan
        from game_core import PLANT_REQUIRED_TICKS
        from touyama_v3.tv3_attacker_route_planner import AttackPlan
        holder=next((a for a in snapshot.allies if a.alive and a.has_spike),None)
        if (holder is None or holder.position not in encoder.scenario.sites[route.site]
                or snapshot.round_timer < PLANT_REQUIRED_TICKS-holder.plant_progress):
            return None
        self.base.plan=AttackPlan(route,mode,'plant',False,'public_plant_time_feasible',holder.position,{})
        return self.base.plan  # No forced PLANT: encoder/policy still select from legal actions.


class RouteProgressProbe:
    def __init__(self, base):
        self.base=base
        self.path=()
        self.best_index=-1
        self.last_progress=0

    def __getattr__(self,name):
        return getattr(self.base,name)

    @property
    def progress_tick(self):
        return self.last_progress

    @progress_tick.setter
    def progress_tick(self,value):
        pass  # Combat wait/repeated retreat cannot reset this diagnostic clock.

    def update(self,snapshot,observation,encoder,model,route=None,mode=None,**kwargs):
        holder=next((a for a in snapshot.allies if a.alive and a.has_spike),None)
        origin=holder.position if holder else snapshot.spike_dropped
        if origin in self.path and self.path.index(origin)>self.best_index:
            self.best_index=self.path.index(origin)
            self.last_progress=snapshot.tick
        plan=self.base.update(snapshot,observation,encoder,model,route,mode,
                              progress_tick_override=self.last_progress,stall_ticks=ROUTE_STALL_TICKS,**kwargs)
        if plan is not None:
            old_suffix=self.path[self.path.index(origin):] if origin in self.path else ()
            new_suffix=plan.route.cells[plan.route.cells.index(origin):] if origin in plan.route.cells else plan.route.cells
            if old_suffix!=new_suffix:
                self.path=new_suffix
                self.best_index=0
                self.last_progress=snapshot.tick
        return plan


def install_probe(controller):
    controller.encoder=ExposureEncoderProbe(controller.encoder)
    if ENABLE_PROGRESS_OVERRIDE:
        controller.route_planner=RouteProgressProbe(controller.route_planner)
    if KEEP_FEASIBLE_PLANT_PLAN:
        controller.route_planner=PlantDeadlineProbe(controller.route_planner)
