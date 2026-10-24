"""Read-only analysis ranking audit. Compare logged and unlogged executions; never train."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
BEST_DIRECTORY=HERE/'data'/'best'
OUTPUT_DIRECTORY=HERE/'logs'/'attacker_route_risk_review'
CASE_SEED=804000042
CASE_ROUND=6
POLICY_MODE='frozen'  # frozen / support_teacher
CHECK_OBSERVER_PARITY=True
AUDIT_ENTRY_ALTERNATIVES=True  # Score legal entry paths without inserting them into the planner.
TORCH_THREADS=1
RISK_SENSITIVITY_TICKS=(33,42,45)  # Feature ablation only; never passed back to the planner.
COMPARISON_ANALYSIS_PATH=None  # Optional existing checkpoint with exactly matching inputs.
HYPOTHETICAL_RIGHT_CONTACTS=((4,(2,40)),(3,(7,38)),(2,(9,40)))
HYPOTHETICAL_CONTACT_TICK=33  # Public information experiment only, not hidden enemy truth.

import argparse
import hashlib
import json
import random
import copy
from dataclasses import replace
from frc_v1.perception import Sighting
from unittest.mock import patch
import numpy as np
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_train_attacker_plant import rollout
from touyama_v3.tv3_learn_attacker_analysis import (
    ROUTE_PRESSURE_WEIGHT,ENTRY_PRESSURE_WEIGHT,Route,shortest_path,branch_sequence,TIME_MARGIN_TICKS)
from touyama_v3.tv3_learn_attacker_fire_support import public_threats,shot_clear
from touyama_v3.tv3_review_attacker_exposure_teacher import incoming_clear
from touyama_v3.tv3_review_attacker_right import candidate_row,json_default,write_row


def risk_row(encoder,snapshot,candidate,placement):
    route=candidate['route']
    features=dict(zip(encoder.route_fields,encoder.route_features(snapshot,route,candidate['mode'],placement)))
    threats=public_threats(snapshot,encoder.history.tracks)
    holder=next((a for a in snapshot.allies if a.alive and a.has_spike),None)
    points=[]
    if holder:
        for index,p in enumerate(route.cells):
            sources=[s for s in threats if incoming_clear(encoder.scenario,snapshot,holder,p,s)]
            if sources:
                points.append(dict(index=index,position=p,enemy_ids=[s.enemy_id for s in sources],
                    asymmetric_enemy_ids=[s.enemy_id for s in sources if not shot_clear(
                        encoder.scenario,snapshot,p,s.position,holder.slot)]))
    weights=dict(losses=.25*candidate['expected_losses']/5,
        damage=.15*candidate['expected_hp_damage']/500,
        abilities=.05*candidate['expected_ability_uses']/10,
        ticks=.05*candidate['expected_plant_ticks_if_success']/100,
        route_region=ROUTE_PRESSURE_WEIGHT*candidate['route_pressure'],
        entry_region=ENTRY_PRESSURE_WEIGHT*candidate['entry_pressure'])
    reconstructed=candidate['plant_probability']-sum(weights.values())
    if not np.isclose(reconstructed,candidate['score'],atol=1e-7):
        raise RuntimeError('Ranking score decomposition mismatch')
    return dict(**candidate_row(candidate),features=features,score_penalties=weights,
                potential_incoming_points=points,
                risk_scope='public_known_positions_and_stationary_allies_not_future_shot_guarantees')


def entry_routes(scenario,origin,remaining_ticks):
    """Diagnostic routes to each real site entrance; preserve differing cell geometry."""
    from game_core import PLANT_REQUIRED_TICKS
    routes={}
    for side in ('L','R'):
        goals=set(scenario.sites[side])
        blocked=goals|set(scenario.sites['R' if side=='L' else 'L'])
        for inside in sorted(goals):
            for outside in scenario.neighbors(inside):
                if outside in blocked:
                    continue
                prefix=shortest_path(scenario,origin,(outside,),blocked)
                cells=prefix+(inside,) if prefix else ()
                if cells and len(cells)-1+PLANT_REQUIRED_TICKS+TIME_MARGIN_TICKS<=remaining_ticks:
                    route=Route(side,cells,branch_sequence(scenario,cells))
                    routes[route.key]=route
    return tuple(routes.values())


class Complete(Exception):
    pass


class RankingAudit:
    def __init__(self,stream,enabled,comparison=None):
        self.stream,self.enabled=stream,enabled
        self.planning=False
        self.pending=None
        self.frames=[]
        self.comparison=comparison

    def starting(self,controller,round_number):
        if not self.enabled or round_number!=CASE_ROUND:
            return
        original=controller.route_planner.update
        def update(*args,**kwargs):
            self.planning=True
            try:
                return original(*args,**kwargs)
            finally:
                self.planning=False
        controller.route_planner.update=update

    def analyze(self,original,encoder,snapshot,observation,candidates=None):
        result=original(encoder,snapshot,observation,candidates)
        if self.planning:
            alive={e.enemy_id for e in snapshot.enemies if e.alive}
            valid={s.enemy_id for s in public_threats(snapshot,encoder.history.tracks)}
            holder=next((a for a in snapshot.allies if a.alive and a.has_spike),None)
            alternatives=[]
            if AUDIT_ENTRY_ALTERNATIVES and holder:
                keys={c['route'].key for c in result['candidates']}
                extra=[r for r in entry_routes(encoder.scenario,holder.position,snapshot.round_timer) if r.key not in keys]
                if extra:
                    alternative_result=original(encoder,snapshot,observation,extra)
                    alternatives=[risk_row(encoder,snapshot,c,alternative_result['placement'])
                                  for c in alternative_result['candidates']]
            self.pending=dict(event='ranking',tick=snapshot.tick,timer=snapshot.round_timer,
                sightings=[dict(enemy_id=s.enemy_id,position=s.position,source=s.source) for s in snapshot.sightings],
                tracks=[dict(enemy_id=i,position=p,age=snapshot.tick-t,
                             alive=i in alive,valid_combat_memory=i in valid,
                             visible_empty=p in snapshot.visible_cells and p not in {s.position for s in snapshot.sightings},
                             region=encoder.scenario.names[int(encoder.scenario.region[p])],
                             predicted_region_mass=result['placement'][i][encoder.scenario.names[int(encoder.scenario.region[p])]])
                        for i,(p,t,_) in encoder.history.tracks.items()],
                candidates=[risk_row(encoder,snapshot,c,result['placement']) for c in result['candidates']],
                unselected_entry_alternatives=alternatives)
            if snapshot.tick in RISK_SENSITIVITY_TICKS:
                model=original.__self__
                exposure_index=encoder.route_fields.index('known_exposure')
                pairs=result['candidates']
                z=np.stack([encoder.route_features(snapshot,c['route'],c['mode'],result['placement']) for c in pairs])
                x=torch.tensor(np.stack([observation]*len(pairs)),dtype=torch.float32)
                sensitivity=[]
                with torch.inference_mode():
                    for level in (0.,.2,.6):
                        modified=z.copy(); modified[:,exposure_index]=level
                        prediction=model(x,torch.tensor(modified,dtype=torch.float32))[1].sigmoid().numpy()
                        sensitivity.append(dict(known_exposure=level,candidates=[dict(
                            key=c['route'].key,mode=c['mode'],plant_probability=float(p[0]),
                            expected_hp_damage=float(p[1]*500),expected_losses=float(p[2]*5))
                            for c,p in zip(pairs,prediction)]))
                self.pending['risk_feature_ablation']=dict(
                    scope='one_input_replaced_not_coherent_world_or_causal_effect',levels=sensitivity)
                if self.comparison is not None:
                    with torch.inference_mode():
                        prediction=self.comparison(x,torch.tensor(z,dtype=torch.float32))[1].sigmoid().numpy()
                    shared=[]
                    for c,p in zip(pairs,prediction):
                        score=float(p[0]-.25*p[2]-.15*p[1]-.05*p[3]-.05*p[4]
                                    -ROUTE_PRESSURE_WEIGHT*c['route_pressure']-ENTRY_PRESSURE_WEIGHT*c['entry_pressure'])
                        shared.append(dict(key=c['route'].key,site=c['route'].site,branches=c['route'].branches,
                            mode=c['mode'],plant_probability=float(p[0]),score=score))
                    unique={c['route'].key:c['route'] for c in pairs}
                    independent=self.comparison.analyze(encoder,snapshot,observation,list(unique.values()))
                    self.pending['comparison_analysis']=dict(
                        scope='same_public_state_not_executed_counterfactual',
                        shared_original_placement=shared,
                        own_placement=[candidate_row(c) for c in independent['candidates']])
            if snapshot.tick==HYPOTHETICAL_CONTACT_TICK and all(i in alive for i,p in HYPOTHETICAL_RIGHT_CONTACTS):
                hypothetical=replace(snapshot,sightings=tuple(Sighting(i,p,'reveal') for i,p in HYPOTHETICAL_RIGHT_CONTACTS))
                cloned=copy.deepcopy(encoder)
                virtual_observation=cloned.observe(hypothetical,[])
                for i,name in enumerate(encoder.fields):
                    if name.startswith('past_'):
                        virtual_observation[i]=observation[i]
                routes=list({c['route'].key:c['route'] for c in result['candidates']}.values())
                experiments={}
                for name,model in (('adopted',original.__self__),('comparison',self.comparison)):
                    if model is not None:
                        # Use the captured original bound method to avoid the audit wrapper itself.
                        evaluated=(original if name=='adopted' else model.analyze)(cloned,hypothetical,virtual_observation,routes)
                        experiments[name]=[candidate_row(c) for c in evaluated['candidates']]
                self.pending['hypothetical_right_contacts']=dict(
                    scope='hypothetical_public_reveal_not_actual_enemy_positions_or_rollout',
                    contacts=HYPOTHETICAL_RIGHT_CONTACTS,models=experiments)
        return result

    def observe(self,controller):
        if controller.snapshot.round_number!=CASE_ROUND:
            return
        snapshot=controller.snapshot
        actors=[dict(slot=a.slot,position=a.position,hp=a.hp,reveal=a.reveal,carrier=a.has_spike,
                     charges=a.charges,ability=a.ability_name,blind=a.blind,forced_facing=a.forced_facing,
                     goal=x.goal,
                     ability_candidates=[dict(index=j,target=x.actions[j].target,legal=bool(x.mask[j]))
                                         for j in range(40,48)],
                     action=x.actions[i].kind,facing=x.actions[i].facing,target=x.actions[i].target)
                for i,x,a in controller.plans.values()]
        frame=dict(event='decision',tick=snapshot.tick,actors=actors,
                   smoke_cells=snapshot.smoke_cells,
                   effects=[dict(kind=e.kind,phase=e.phase,position=e.position,cells=e.cells,
                                 direction=e.direction) for e in snapshot.effects],
                   route=controller.route.key if controller.route else None,
                   reason=controller.attack_plan.reason if controller.attack_plan else None)
        self.frames.append(frame)
        if self.pending is not None:
            event=controller.route_planner.events[-1] if controller.route_planner.events else {}
            holder=next((a for a in snapshot.allies if a.alive and a.has_spike),None)
            origin=holder.position if holder else snapshot.spike_dropped
            selected=controller.route
            if selected is not None and origin in selected.cells:
                cells=selected.cells[selected.cells.index(origin):]
                selected=Route(selected.site,cells,branch_sequence(controller.scenario,cells))
            self.pending.update(selected_key=selected.key if selected is not None else None,
                                selected_full_route_key=controller.route.key if controller.route else None,
                                selected_mode=controller.route_mode,
                                planner_event=event,planner_gate=controller.route_planner.decision_debug)
            write_row(self.stream,self.pending)
            self.pending=None
        write_row(self.stream,frame)


def main():
    global CASE_SEED,CASE_ROUND,POLICY_MODE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=CASE_SEED)
    parser.add_argument('--round',type=int,default=CASE_ROUND)
    parser.add_argument('--mode',choices=('frozen','support_teacher'),default=POLICY_MODE)
    parser.add_argument('--compare-analysis',type=Path,default=COMPARISON_ANALYSIS_PATH)
    args=parser.parse_args()
    CASE_SEED,CASE_ROUND,POLICY_MODE=args.seed,args.round,args.mode
    if not 1<=CASE_ROUND<=12:
        raise ValueError('Round must be in 1..12')
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    comparison=None
    comparison_hash=None
    if args.compare_analysis is not None:
        from touyama_v3.tv3_learn_attacker_analysis import load_analysis,AttackerEncoder
        comparison,comparison_encoder,_=load_analysis(args.compare_analysis,scenario,OPPONENT)
        original_encoder=AttackerEncoder(scenario,source['analysis'].execution_version)
        if comparison_encoder.fields!=original_encoder.fields or comparison_encoder.route_fields!=original_encoder.route_fields:
            raise ValueError('Comparison input contract differs; disable comparison or use a matching checkpoint')
        comparison_hash=hashlib.sha256(args.compare_analysis.read_bytes()).hexdigest()
    config=load_checkpoint(source['paths']['plant'])['config']
    if POLICY_MODE=='support_teacher':
        from touyama_v3.tv3_learn_attacker_plant import PlantDQN
        policy=PlantDQN(11).eval().requires_grad_(False)
    else:
        policy=source['plant']
    OUTPUT_DIRECTORY.mkdir(parents=True,exist_ok=True)
    original=source['analysis'].analyze
    executions=[]
    for enabled in ((False,True) if CHECK_OBSERVER_PARITY else (True,)):
        rounds=[]
        trace=OUTPUT_DIRECTORY/f'{POLICY_MODE}_{CASE_SEED}_r{CASE_ROUND}_{enabled}.jsonl'
        with trace.open('w',encoding='utf-8') as stream:
            audit=RankingAudit(stream,enabled,comparison)
            def completed(record):
                rounds.append(record)
                if record['round']==CASE_ROUND:
                    raise Complete()
            def analyze(*values,**kwargs):
                return audit.analyze(original,*values,**kwargs)
            with patch.object(source['analysis'],'analyze',side_effect=analyze):
                try:
                    rollout(OPPONENT,scenario,source['analysis'],policy,config,CASE_SEED,
                        preset_name='Touyama Gaming',training=POLICY_MODE!='frozen',
                        teacher_probability=1. if POLICY_MODE!='frozen' else 0.,
                        round_start_callback=audit.starting,tick_callback=audit.observe,round_callback=completed)
                except Complete:
                    pass
            executions.append(dict(rounds=rounds,frames=audit.frames,
                python_rng=repr(random.getstate()),numpy_rng=repr(np.random.get_state())))
    if CHECK_OBSERVER_PARITY and executions[0]!=executions[1]:
        raise RuntimeError('Read-only ranking observer changed execution/state/RNG')
    result=dict(scope='read_only_ranking_audit_not_retraining',seed=CASE_SEED,round=CASE_ROUND,
        mode=POLICY_MODE,observer_parity=CHECK_OBSERVER_PARITY,optimizer_updates=0,
        source_hashes=source['hashes'],scenario=scenario.signature,
        comparison_analysis_hash=comparison_hash,
        record=executions[-1]['rounds'][-1],trace=str(trace),
        script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output=OUTPUT_DIRECTORY/f'{POLICY_MODE}_{CASE_SEED}_r{CASE_ROUND}_summary.json'
    output.write_text(json.dumps(result,default=json_default,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',output,'parity:',CHECK_OBSERVER_PARITY,'outcome:',result['record']['reason'])


if __name__=='__main__':
    main()
