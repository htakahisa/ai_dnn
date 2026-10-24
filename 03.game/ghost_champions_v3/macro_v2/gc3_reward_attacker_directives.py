"""Instruction credit plus the retained retrieval/plant/combat/actual-win rewards."""
from collections import Counter
import numpy as np
from grid_paths import distance_map
from ..preplant.gc3_rollout_attacker_preplant import PreplantRewards,SITE_PROGRESS_WEIGHT,DISTANCE_SCALE,preplant_reward_settings

REWARD_VERSION='gc3-micro-directives-spawn-retreat-v2'


def reward_settings(config):
    return dict(preplant_reward_settings(),directive_reward_version=REWARD_VERSION,
        ORDER_APPROACH_WEIGHT=config['ORDER_APPROACH_WEIGHT'],WAIT_HOLD_REWARD=config['WAIT_HOLD_REWARD'],
        WAIT_MOVE_PENALTY=config['WAIT_MOVE_PENALTY'],
        ORDER_AWAY_PENALTY=config['ORDER_AWAY_PENALTY'],ORDER_ARRIVAL_REWARD=config['ORDER_ARRIVAL_REWARD'],
        order_targets='ADVANCE=user candidate; RETREAT=own actual round-start spawn; WAIT=hold',
        planned_plant_progress='distance to the fixed planned plant coordinate; no target-switch bonus')


class DirectiveRewards(PreplantRewards):
    def __init__(self,game,gamma,controller,recorder,config):
        self.controller,self.recorder,self.config=controller,recorder,config
        self.order_best={};self.order_arrivals=set();self.distances={}
        self.measures=Counter()
        super().__init__(game,gamma,lambda:controller.current_goal)

    def distance(self,target,position):
        target=tuple(target)
        if target not in self.distances:self.distances[target]=distance_map(self.game.grid,target)
        return int(self.distances[target][tuple(position)])

    def site_potential(self,position,goal):
        if goal is None or self.controller.current_plan is None:return 0.
        distance=self.distance(self.controller.current_plan['plant'],position)
        return SITE_PROGRESS_WEIGHT/(1.+distance/DISTANCE_SCALE) if distance>=0 else 0.

    def begin_tick(self):
        self.before_positions={str(c.name):tuple(c.pos) for c in self.game.chars if c.team=='A' and c.is_alive}
        self.before_indices={name:row for name,row in self.recorder.latest.items()}
        super().begin_tick()

    def finish_tick(self,terminal,success):
        shared,personal=super().finish_tick(terminal,success)
        for name,position in self.before_positions.items():
            row=self.recorder.latest.get(name)
            if row is None or row is self.before_indices.get(name):continue
            directive=row['macro_directive'];instruction=directive['instruction']
            unit=self.own[name]
            after=tuple(unit.pos)
            parts=personal.setdefault(name,{})
            if instruction=='WAIT':
                # Failed attempts to move must not masquerade as obedient holds.
                mode=row['selections']['mode']
                moved_choice=mode in (0,2) and row['selections'].get('position')!=np.ravel_multi_index(position,self.game.grid.shape)
                held=after==position and not moved_choice
                parts['order_wait']=self.config['WAIT_HOLD_REWARD'] if held else -self.config['WAIT_MOVE_PENALTY']
                self.measures['WAIT_total']+=1;self.measures['WAIT_satisfied']+=int(held)
                self.measures['WAIT_applicable']+=1;self.measures['WAIT_compliant']+=int(held)
                continue
            target=directive['retreat_target'] if instruction=='RETREAT' else directive['destination']
            key=name,instruction,tuple(target)
            before_dist,after_dist=self.distance(target,position),self.distance(target,after)
            self.measures[instruction+'_total']+=1
            satisfied=after_dist>=0 and after_dist<before_dist or before_dist==0 and after_dist==0
            self.measures[instruction+'_satisfied']+=int(satisfied)
            # Measure approach only where movement is actually possible, not
            # retreat commands issued while already standing in spawn.
            masks=row.get('masks',{})
            legal=masks.get('position')
            if legal is None:legal=(self.game.grid!=1).reshape(-1)
            distance_grid=self.distances[tuple(target)]
            improving=np.asarray(legal).reshape(self.game.grid.shape)&(distance_grid>=0)&(distance_grid<before_dist)
            modes=masks.get('mode')
            objective=modes is not None and (modes[5] or modes[7])
            applicable=bool(before_dist>0 and improving.any() and not objective)
            self.measures[instruction+'_applicable']+=int(applicable)
            self.measures[instruction+'_compliant']+=int(applicable and satisfied)
            self.measures[instruction+'_arrivals']+=int(before_dist>0 and after_dist==0)
            near=lambda d:self.config['ORDER_APPROACH_WEIGHT']/(1.+d/DISTANCE_SCALE) if d>=0 else 0.
            before_near,after_near=near(before_dist),near(after_dist)
            best=max(self.order_best.get(key,0.),before_near)
            component='order_advance' if instruction=='ADVANCE' else 'order_retreat'
            parts[component]=max(0.,after_near-best)
            if before_dist>=0 and after_dist>before_dist:
                parts[component+'_away']=-self.config['ORDER_AWAY_PENALTY']*(after_dist-before_dist)
            if before_dist>0 and after_dist==0 and key not in self.order_arrivals:
                parts[component+'_arrival']=self.config['ORDER_ARRIVAL_REWARD']
                self.order_arrivals.add(key)
            self.order_best[key]=max(best,after_near)
        return shared,personal
