"""Expose stable, sampled commands while training only the micro instruction inputs."""
from copy import deepcopy
import random
import numpy as np
from ..preplant.gc3_observation_attacker_history import encode_public,EXTRA_WIDTH
from grid_paths import distance_map


class SimpleDirectiveLogic:
    """One site per round, a plant destination and nearby team destinations."""
    def __init__(self,seed,retreat_ticks=0,retreat_after_ticks=12,retreat_min_distance=6,retreat_wait_ticks=2,
                 retreat_max_distance=None):
        if min(retreat_ticks,retreat_after_ticks,retreat_wait_ticks)<0 or retreat_min_distance<1 or (
                retreat_max_distance is not None and retreat_max_distance<retreat_min_distance):
            raise ValueError('Invalid simple retreat practice settings')
        self.seed=seed
        self.retreat_ticks=retreat_ticks;self.retreat_after_ticks=retreat_after_ticks
        self.retreat_min_distance=retreat_min_distance;self.retreat_wait_ticks=retreat_wait_ticks
        self.retreat_max_distance=retreat_max_distance
        self.reset_round()

    def reset_round(self):
        self.plan=None;self.points=None
        self.spawn_positions={};self.spawn_distances={};self.battle_age=0
        self.retreated=set();self.active_retreat=None;self.pause_until={};self.retreat_events=[]

    @staticmethod
    def busy_with_objective(unit,grid):
        return bool(getattr(unit,'plant_timer',0)>0 or getattr(unit,'orb_collect_timer',0)>0 or
            getattr(unit,'has_spike',False) and grid[tuple(unit.pos)]==2)

    def practice_retreat(self,own,state,instructions):
        if state.get('defender_setup_active') or state.get('is_planted'):return
        grid=state['grid'];by_name={str(unit.name):unit for unit in own}
        if self.active_retreat is not None:
            event=self.active_retreat;unit=by_name.get(event['name'])
            reason=None
            if unit is None or not unit.is_alive:reason='died'
            elif self.busy_with_objective(unit,grid):reason='objective'
            elif tuple(unit.pos)==self.spawn_positions[event['name']]:reason='reached_spawn'
            elif self.battle_age-event['start']>=self.retreat_ticks:reason='timeout'
            if reason is not None:
                event.update(end=self.battle_age,reason=reason)
                # Resume planting immediately if it becomes available.
                if reason not in ('died','objective'):
                    self.pause_until[event['name']]=self.battle_age+self.retreat_wait_ticks
                self.active_retreat=None
        if self.retreat_ticks and self.active_retreat is None and self.battle_age>=self.retreat_after_ticks:
            # One player at a time, once per round; never send the whole team back.
            offset=self.seed%max(1,len(own))
            ordered=own[offset:]+own[:offset]
            for unit in ordered:
                name=str(unit.name)
                if not unit.is_alive or name in self.retreated or self.busy_with_objective(unit,grid):continue
                distance=int(self.spawn_distances[name][tuple(unit.pos)])
                if distance<self.retreat_min_distance:continue
                if self.retreat_max_distance is not None and distance>self.retreat_max_distance:continue
                event=dict(name=name,start=self.battle_age,start_distance=distance)
                self.retreat_events.append(event);self.active_retreat=event;self.retreated.add(name)
                break
        for slot,unit in enumerate(own):
            if not unit.is_alive:continue
            name=str(unit.name)
            if self.active_retreat is not None and name==self.active_retreat['name']:
                instructions[slot]='RETREAT'
            elif self.battle_age<self.pause_until.get(name,0) and not self.busy_with_objective(unit,grid):
                instructions[slot]='WAIT'
        self.battle_age+=1

    def select_macro(self,char,state,history,current_plan,candidates):
        candidates.validate_arena(state['grid'])
        own=sorted((c for c in state['chars'] if c.team=='A'),key=lambda c:str(c.name))[:5]
        active=np.array([i<len(own) and own[i].is_alive for i in range(5)],dtype=bool)
        if self.plan is None:
            self.spawn_positions={str(unit.name):tuple(map(int,unit.pos)) for unit in own}
            self.spawn_distances={name:distance_map(state['grid'],point) for name,point in self.spawn_positions.items()}
            points=[candidates.coordinate(i) for i in np.flatnonzero(candidates.destination_mask)]
            plants=[candidates.coordinate(i) for i in np.flatnonzero(candidates.plant_mask)]
            # Prefer a plant cell that is also a user destination: the carrier's
            # two orders then agree exactly, without inventing map candidates.
            sites=[[p for p in plants if int(p[1]>=state['grid'].shape[1]//2)==site] for site in (0,1)]
            site=self.seed%2
            if not sites[site]:site=1-site
            pool=[p for p in sites[site] if p in points] or sites[site]
            if not pool:raise ValueError('Simple macro requires legal plant coordinates')
            plant=random.Random(self.seed).choice(pool)
            distances=distance_map(state['grid'],plant)
            self.points=sorted((p for p in points if distances[p]>=0),key=lambda p:(int(distances[p]),p))
            if not self.points:raise ValueError('No destination candidate is reachable from the planned plant')
            self.plan=dict(destinations=[None]*5,instructions=[None]*5,plant=plant)
        carrier=next((i for i,c in enumerate(own) if c.is_alive and getattr(c,'has_spike',False)),None)
        available=self.points[1:] or self.points
        destinations=[None]*5;instructions=[None]*5;escort=0
        for slot,unit in enumerate(own):
            if not active[slot]:continue
            if slot==carrier:
                destinations[slot]=self.points[0];instructions[slot]='ADVANCE'
            else:
                destinations[slot]=available[escort%len(available)];escort+=1
                instructions[slot]='WAIT' if tuple(unit.pos)==destinations[slot] else 'ADVANCE'
        self.practice_retreat(own,state,instructions)
        plan=dict(destinations=destinations,instructions=instructions,plant=self.plan['plant'])
        observation=encode_public(char,state,history);observation[-EXTRA_WIDTH+6]=2.
        return plan,dict(obs=observation,roster=[str(c.name) for c in own],active=active)


def directive_policy(config,seed):
    mode=config.get('MACRO_DIRECTIVE_MODE','random')
    if mode=='simple':return SimpleDirectiveLogic(seed,config.get('SIMPLE_RETREAT_TICKS',0),
        config.get('SIMPLE_RETREAT_AFTER_TICKS',12),config.get('SIMPLE_RETREAT_MIN_DISTANCE',6),
        config.get('SIMPLE_RETREAT_WAIT_TICKS',2),config.get('SIMPLE_RETREAT_MAX_DISTANCE'))
    if mode!='random':raise ValueError('MACRO_DIRECTIVE_MODE must be simple or random')
    return DirectiveSampler(seed,config['ORDER_DURATION_TICKS'],config['ORDER_PROBABILITIES'],config.get('ORDER_WARMUP_TICKS',0))


class DirectiveSampler:
    def __init__(self,seed,duration,probabilities,warmup_ticks=0):
        if duration<1 or warmup_ticks<0 or len(probabilities)!=3 or min(probabilities)<0 or not np.isclose(sum(probabilities),1.):
            raise ValueError('Invalid instruction duration/probabilities')
        self.seed,self.duration,self.probabilities=seed,duration,tuple(probabilities)
        self.warmup_ticks=warmup_ticks
        self.reset_round()

    def reset_round(self):
        self.rng=random.Random(self.seed)
        self.tick=0;self.plan=None

    def select_macro(self,char,state,history,current_plan,candidates):
        candidates.validate_arena(state['grid'])
        own=sorted((c for c in state['chars'] if c.team=='A'),key=lambda c:str(c.name))[:5]
        active=np.array([i<len(own) and own[i].is_alive for i in range(5)],dtype=np.bool_)
        points=[candidates.coordinate(i) for i in np.flatnonzero(candidates.destination_mask)]
        if self.plan is None:
            plants=[candidates.coordinate(i) for i in np.flatnonzero(candidates.plant_mask)]
            sites=[[p for p in plants if int(p[1]>=state['grid'].shape[1]//2)==site] for site in (0,1)]
            available=[site for site in sites if site]
            plant=self.rng.choice(self.rng.choice(available))
            self.plan=dict(destinations=[None]*5,instructions=[None]*5,plant=plant)
        update=self.tick==0 or self.tick==self.warmup_ticks or (
            self.tick>self.warmup_ticks and (self.tick-self.warmup_ticks)%self.duration==0)
        if update:
            for slot in range(5):
                self.plan['destinations'][slot]=self.rng.choice(points)
                self.plan['instructions'][slot]=self.rng.choices(('WAIT','ADVANCE','RETREAT'),weights=self.probabilities,k=1)[0]
                if self.tick<self.warmup_ticks:self.plan['instructions'][slot]='ADVANCE'
        plan=deepcopy(self.plan)
        for slot in np.flatnonzero(~active):plan['destinations'][slot]=plan['instructions'][slot]=None
        self.tick+=1
        observation=encode_public(char,state,history);observation[-EXTRA_WIDTH+6]=2.
        return plan,dict(obs=observation,roster=[str(c.name) for c in own],active=active)
