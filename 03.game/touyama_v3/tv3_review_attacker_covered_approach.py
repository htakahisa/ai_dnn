"""Diagnostic public covered-route commitment; no learning or production changes."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
BEST_DIRECTORY=HERE/'data'/'best'
OUTPUT_DIRECTORY=HERE/'logs'/'attacker_route_risk_review'
CASE_SEED=804000042
CASE_ROUND=6
INTERVENTION_TICK=33
SITE='R'
POLICY_MODE='frozen'  # frozen / support_teacher / corner_teacher
CHECK_CONTROL=True
TORCH_THREADS=1
OUTPUT_PREFIX='covered_corner_window'
PRIVATE_POST_MOVE_LABELS=True  # Diagnostic labels only, after policy actions freeze; never fed to the controller.

import argparse
import hashlib
import json
import random
import numpy as np
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_train_attacker_plant import rollout
from touyama_v3.tv3_learn_attacker_analysis import Route,shortest_path,branch_sequence,TIME_MARGIN_TICKS
from touyama_v3.tv3_attacker_route_planner import AttackPlan,WAYPOINT_STEPS,ENTRY_STEPS
from touyama_v3.tv3_learn_attacker_fire_support import public_threats
from touyama_v3.tv3_review_attacker_safe_entry import covered_paths,corner_flash_options
from touyama_v3.tv3_review_attacker_right import write_row,json_default
from game_core import PLANT_REQUIRED_TICKS


class Complete(Exception):
    pass


class CoveredRouteProbe:
    def __init__(self,base,stream,enabled,controller,reset_entry=False,upgrade_policy=None):
        self.base,self.stream,self.enabled=base,stream,enabled
        self.controller,self.reset_entry=controller,reset_entry
        self.upgrade_policy=upgrade_policy
        self.committed=None
        self.staging=None
        self.initial_hash=None
        self.intervention=None

    def __getattr__(self,name):
        return getattr(self.base,name)

    @property
    def progress_tick(self):
        return self.base.progress_tick

    @progress_tick.setter
    def progress_tick(self,value):
        self.base.progress_tick=value

    def update(self,snapshot,observation,encoder,model,route=None,mode=None,**kwargs):
        holder=next((a for a in snapshot.allies if a.alive and a.has_spike),None)
        if snapshot.tick>=INTERVENTION_TICK and self.initial_hash is None:
            state=dict(snapshot=repr(snapshot),python_rng=repr(random.getstate()),numpy_rng=repr(np.random.get_state()),
                       controller_rng=self.controller.rng.bit_generator.state,
                       physical_labels=[(str(c.name),c.team,tuple(c.pos),c.hp,c.facing,c.blind_remaining,c.reveal_remaining)
                                        for c in self.controller.game.chars])  # Comparison label only; never selects a route.
            self.initial_hash=hashlib.sha256(json.dumps(state).encode()).hexdigest()
            if self.enabled and holder:
                positions=tuple(s.position for s in public_threats(snapshot,encoder.history.tracks))
                approaches=covered_paths(encoder.scenario,holder.position,SITE,positions) if positions else []
                possible=[]
                for p in approaches:
                    suffix=shortest_path(encoder.scenario,p['staging'],encoder.scenario.sites[SITE],
                                         encoder.scenario.sites['L' if SITE=='R' else 'R'])
                    cells=p['path']+suffix[1:] if suffix else ()
                    if (cells and len(cells)==len(set(cells)) and
                            len(cells)-1+PLANT_REQUIRED_TICKS+TIME_MARGIN_TICKS<=snapshot.round_timer):
                        possible.append((len(cells),p['staging'],cells))
                if possible:
                    _,self.staging,cells=min(possible)
                    self.committed=Route(SITE,cells,branch_sequence(encoder.scenario,cells))
                    if self.reset_entry:
                        self.controller.site_assembly.follow_route_entry=True
                    if self.upgrade_policy is not None:
                        from touyama_v3.tv3_learn_attacker_plant import PlantEncoder
                        old=self.controller.encoder
                        replacement=PlantEncoder(encoder.scenario,12)
                        replacement.experimental_flash_teacher=True  # Deliberate diagnostic, never the training default.
                        replacement.combat_coach,replacement.support_coach=old.combat_coach,old.support_coach
                        self.controller.encoder=replacement
                        self.controller.policy=self.upgrade_policy
                    self.intervention=dict(tick=snapshot.tick,site=SITE,staging=self.staging,
                        reset_entry_context=self.reset_entry,
                        route=cells,branches=self.committed.branches,known_positions=positions,
                        flash_options=corner_flash_options(encoder.scenario,self.staging,SITE,positions,snapshot.smoke_cells))
                    write_row(self.stream,dict(event='intervention',**self.intervention))
                else:
                    self.intervention=dict(tick=snapshot.tick,reason='no_time_feasible_covered_approach')
                    write_row(self.stream,dict(event='intervention',**self.intervention))
        if self.committed is None:
            return self.base.update(snapshot,observation,encoder,model,route,mode,**kwargs)
        cells=self.committed.cells
        index=cells.index(holder.position) if holder and holder.position in cells else 0
        changed=route is None or route.key!=self.committed.key
        phase='entry' if len(cells)-index<=ENTRY_STEPS else 'approach'
        plan=AttackPlan(self.committed,mode or 'supported',phase,changed,'diagnostic_covered_commitment',
                       cells[min(index+WAYPOINT_STEPS,len(cells)-1)],{})
        self.base.flank_entries={}
        if changed:
            self.base.events.append(dict(tick=snapshot.tick,site=SITE,key=self.committed.key,
                                         reason='diagnostic_covered_commitment',branches=self.committed.branches))
        self.base.plan=plan
        return plan


def main():
    global POLICY_MODE,CHECK_CONTROL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('frozen','support_teacher','corner_teacher'),default=POLICY_MODE)
    parser.add_argument('--no-control',action='store_true')
    parser.add_argument('--prefix',default=OUTPUT_PREFIX)
    args=parser.parse_args()
    POLICY_MODE=args.mode
    CHECK_CONTROL=CHECK_CONTROL and not args.no_control
    prefix=args.prefix+('_labels' if args.no_control else '')
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    config=load_checkpoint(source['paths']['plant'])['config']
    policy=source['plant']
    upgraded_policy=None
    if POLICY_MODE in ('support_teacher','corner_teacher'):
        from touyama_v3.tv3_learn_attacker_plant import PlantDQN
        policy=PlantDQN(11).eval().requires_grad_(False)
        if POLICY_MODE=='corner_teacher':
            upgraded_policy=PlantDQN(12).eval().requires_grad_(False)
    OUTPUT_DIRECTORY.mkdir(parents=True,exist_ok=True)
    results=[]
    variants=('control','route_only','route_and_entry','route_entry_corner') if upgraded_policy is not None else ('control','route_only','route_and_entry')
    for variant in (variants if CHECK_CONTROL else (variants[-1],)):
        enabled=variant!='control'
        probe=None
        trace=OUTPUT_DIRECTORY/f'{prefix}_{POLICY_MODE}_{variant}.jsonl'
        with trace.open('w',encoding='utf-8') as stream:
            labels_installed=False
            def starting(controller,round_number):
                nonlocal probe
                if round_number==CASE_ROUND:
                    probe=CoveredRouteProbe(controller.route_planner,stream,enabled,controller,
                        variant in ('route_and_entry','route_entry_corner'),
                        upgraded_policy if variant=='route_entry_corner' else None)
                    controller.route_planner=probe
            def observe(controller):
                nonlocal labels_installed
                snap=controller.snapshot
                if snap.round_number!=CASE_ROUND:
                    return
                if PRIVATE_POST_MOVE_LABELS and not labels_installed:
                    game=controller.game
                    original=game._resolve_all_shots
                    def shots(*values,**options):
                        if game.current_round==CASE_ROUND and not game.is_planted:
                            write_row(stream,dict(event='post_move_labels',tick=game.battle_tick,
                                scope='private_reward_diagnostic_not_policy_input',
                                chars=[dict(name=str(c.name),team=c.team,position=tuple(c.pos),
                                    alive=c.is_alive,hp=c.hp,blind=c.blind_remaining,reveal=c.reveal_remaining)
                                    for c in game.chars]))
                        return original(*values,**options)
                    game._resolve_all_shots=shots
                    labels_installed=True
                write_row(stream,dict(event='decision',tick=snap.tick,
                    actors=[dict(slot=a.slot,position=a.position,hp=a.hp,carrier=a.has_spike,
                        ability=a.ability_name,charges=a.charges,action=x.actions[i].kind,
                        target=x.actions[i].target,combat=x.combat.reason,goal=x.goal)
                        for i,x,a in controller.plans.values()],
                    sightings=[dict(enemy_id=s.enemy_id,position=s.position,source=s.source) for s in snap.sightings],
                    effects=[dict(kind=e.kind,phase=e.phase,position=e.position) for e in snap.effects]))
            def completed(record):
                if record['round']==CASE_ROUND:
                    results.append(dict(variant=variant,enabled=enabled,initial_state_hash=probe.initial_hash,
                        execution_version=probe.controller.encoder.execution_version,
                        experimental_flash_window_teacher=probe.controller.encoder.experimental_flash_teacher,
                        intervention=probe.intervention,record=record,trace=str(trace)))
                    raise Complete()
            try:
                rollout(OPPONENT,scenario,source['analysis'],policy,config,CASE_SEED,preset_name='Touyama Gaming',
                    training=POLICY_MODE!='frozen',teacher_probability=1. if POLICY_MODE!='frozen' else 0.,
                    round_start_callback=starting,tick_callback=observe,round_callback=completed)
            except Complete:
                pass
    if CHECK_CONTROL and len({r['initial_state_hash'] for r in results})!=1:
        raise RuntimeError('Pre-intervention public state/RNG mismatch')
    output=OUTPUT_DIRECTORY/f'{prefix}_{POLICY_MODE}_summary.json'
    output.write_text(json.dumps(dict(scope='diagnostic_route_commitment_not_learned_performance',
        seed=CASE_SEED,round=CASE_ROUND,source_hashes=source['hashes'],scenario=scenario.signature,
        optimizer_updates=0,results=results),default=json_default,ensure_ascii=False,indent=2),encoding='utf-8')
    print([(r['variant'],r['record']['reason'],r['record']['alive'],r['intervention'] is not None) for r in results])
    print('Saved:',output)


if __name__=='__main__':
    main()
