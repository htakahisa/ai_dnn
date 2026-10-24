"""Focused public trace of one real timeout; no optimizer or training."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
OPPONENT='frc_v1'
SOURCE_DIRECTORY=HERE/'data'/'best'
SOURCE_PLANT_CHECKPOINT=None  # Optional explicit evaluation checkpoint; schema/dependencies remain strict.
EXPECTED_PLANT_SET=10
SEED=804000142
TARGET_ROUND=8
START_ROUND=1  # Independent evaluation blocks begin at either 1 or 13.
OUTPUT_PATH=HERE/'logs'/'attacker_route_risk_review'/'set10_r8_timeout_decisions.json'
TORCH_THREADS=1
COMBAT_DIAGNOSTICS=True  # Private post-action labels remain outside model inputs.
RESTORE_UNIQUE_PROGRESS_EDGE=False  # Diagnostic only; never changes production masks.
DEFER_INITIAL_ASSEMBLY=False  # Measure distance to the selected outside entry before rallying.
RESTORE_ALL_ROUTE_PROGRESS_EDGES=False  # Diagnostic: also recover forward motion when a backward move is legal.
NET_PROGRESS_CLOCK=False  # Diagnostic: best fixed plant distance, rather than any movement.
CONFIRMED_EMPTY_VIEW=False  # Diagnostic: require consecutive public visibility before rejecting a track.
CANDIDATE_VERSION=None  # Copy weights in memory for a versioned encoder experiment only.
CANDIDATE_PLANT_CHECKPOINT=None  # Switch to validated learned weights at the target round only.
CANDIDATE_EXPECTED_SET=5
CANDIDATE_TARGET_ONLY=True  # Match all preceding rounds before changing the encoder.
FORCE_DIAGNOSTIC_SITE=None  # Counterfactual evaluation only; never production site selection.
FORCE_TEACHER_TARGET=False  # Read-only teacher rollout; no optimizer or checkpoint saving.
RESTORE_CONFIRMED_FLASH_PROGRESS=False  # Diagnostic: only a progress edge covered by owned public flash.
REASSIGN_BLOCKED_PLANT_HELPER=False  # Diagnostic goal change; model still selects the action.
RELEASE_DEADLINE_ROUTE_RESERVE=False  # Only when the normal candidate list is empty.
DIAGNOSTIC_CHOICE_OVERRIDES={}  # {(tick, slot): legal action index}; counterfactual only.
DIAGNOSTIC_PLAN_ORDER_OVERRIDES={}  # {tick: (first slot, ...)}; public reservation-order counterfactual.

import json
import hashlib
import random
import pickle
import numpy as np
import torch
from unittest.mock import patch
from frc_v1.actions import MOVE_STEPS
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources, read_checkpoint
from touyama_v3.tv3_train_attacker_plant import rollout
from touyama_v3.tv3_learn_attacker_fire_support import public_threats, shot_clear, target_support
from touyama_v3.tv3_learn_attacker_recon_cover import incoming_clear
from touyama_v3.tv3_attacker_route_planner import AdaptiveAttackPlanner
from touyama_v3.tv3_learn_attacker_plant import PlantEncoder,PlantDQN
from dataclasses import replace
from touyama_v3.tv3_attacker_threat_evidence import planned_team_snapshot
from touyama_v3.tv3_attacker_site_entry import traffic_distances,MAX_TRAFFIC_DETOUR_STEPS
from grid_paths import distance_map
from touyama_v3 import tv3_learn_attacker_analysis as route_analysis


class TargetFinished(Exception):
    pass


def json_value(value):
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,set):
        return sorted(value)
    raise TypeError(f'Unsupported diagnostic value: {type(value).__name__}')


def load_review_source(scenario,plant_checkpoint=None):
    source=load_sources(scenario,(OPPONENT,),SOURCE_DIRECTORY,SOURCE_DIRECTORY)[OPPONENT]
    checkpoint=SOURCE_PLANT_CHECKPOINT if plant_checkpoint is None else plant_checkpoint
    if checkpoint is not None:
        from touyama_v3.tv3_learn_attacker_plant import policy_schema
        state,digest=read_checkpoint(checkpoint)
        version=state['schema'].get('version')
        from touyama_v3.tv3_model_extensions import load_extensions
        custom=load_extensions(state['schema'].get('custom_extensions'),role='attacker_plant',opponent=OPPONENT)
        if state['opponent']!=OPPONENT:
            raise ValueError('Diagnostic checkpoint opponent mismatch')
        if json.dumps(state['schema'],sort_keys=True)!=json.dumps(policy_schema(scenario,version,custom_extensions=custom.contract,weight_routing=state['schema'].get('weight_routing')),sort_keys=True):
            raise ValueError('Diagnostic checkpoint map/schema mismatch')
        if state['analysis_hash']!=source['hashes']['analysis']:
            raise ValueError('Diagnostic checkpoint analysis dependency mismatch')
        plant=PlantDQN(version)
        plant.opponent=OPPONENT
        plant.custom_extension_contract=custom.contract
        plant.load_state_dict(state['model'])
        from touyama_v3.tv3_learn_attacker_plant import restore_policy_routing
        restore_policy_routing(plant,state,scenario)
        plant.eval()
        plant.requires_grad_(False)
        source['plant']=plant
        source['plant_set']=state['completed_sets']
        source['hashes']['plant']=digest
        source['paths']['plant']=str(Path(checkpoint).resolve())
    return source


def main():
    if not START_ROUND<=TARGET_ROUND<START_ROUND+12:
        raise ValueError('Target round must belong to the declared 12-round evaluation block')
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_review_source(scenario)
    if source['plant_set']!=EXPECTED_PLANT_SET:
        raise ValueError(f"Expected set{EXPECTED_PLANT_SET}, found set{source['plant_set']}; do not mislabel a different model")
    policy=source['plant']
    candidate=None
    if CANDIDATE_PLANT_CHECKPOINT is not None:
        if CANDIDATE_VERSION is not None:
            raise ValueError('Specify candidate checkpoint or copied-weight version, not both')
        candidate=load_review_source(scenario,CANDIDATE_PLANT_CHECKPOINT)
        if candidate['plant_set']!=CANDIDATE_EXPECTED_SET:
            raise ValueError('Unexpected candidate checkpoint set')
        policy=candidate['plant']
    if CANDIDATE_VERSION is not None:
        policy=PlantDQN(CANDIDATE_VERSION)
        policy.load_state_dict(source['plant'].state_dict())
        policy.eval()
    output=dict(seed=SEED,round=TARGET_ROUND,source_hashes=source['hashes'],plant_set=source['plant_set'],
        optimizer_updates=0,restore_unique_progress_edge=RESTORE_UNIQUE_PROGRESS_EDGE,
        defer_initial_assembly=DEFER_INITIAL_ASSEMBLY,restore_all_progress_edges=RESTORE_ALL_ROUTE_PROGRESS_EDGES,frames=[])
    output['source_paths']=source['paths']
    output['source_version']=source['plant'].execution_version
    output['start_round']=START_ROUND
    output['net_progress_clock']=NET_PROGRESS_CLOCK
    output['confirmed_empty_view']=CONFIRMED_EMPTY_VIEW
    output['candidate_version']=CANDIDATE_VERSION
    output['candidate_checkpoint']=None if candidate is None else dict(
        path=candidate['paths']['plant'],hashes=candidate['hashes'],
        version=policy.execution_version,plant_set=candidate['plant_set'])
    from touyama_v3.tv3_learn_attacker_plant import policy_schema
    output['execution_schema_hash']=hashlib.sha256(json.dumps(
        policy_schema(scenario,policy.execution_version,custom_extensions=getattr(policy,'custom_extension_contract',None),
            weight_routing=getattr(policy,'weight_routing',None)),sort_keys=True).encode()).hexdigest()
    output['candidate_target_only']=CANDIDATE_TARGET_ONLY
    output['preceding_rounds']=[]
    output['forced_diagnostic_site']=FORCE_DIAGNOSTIC_SITE
    output['forced_teacher_target']=FORCE_TEACHER_TARGET
    output['restore_confirmed_flash_progress']=RESTORE_CONFIRMED_FLASH_PROGRESS
    output['reassign_blocked_plant_helper']=REASSIGN_BLOCKED_PLANT_HELPER
    output['release_deadline_route_reserve']=RELEASE_DEADLINE_ROUTE_RESERVE
    output['choice_overrides']=[]
    output['plan_order_overrides']=[]
    original_prepare=TouyamaV3AttackerPlantController.prepare_team_tick
    applied_choices=set()
    def prepare(c):
        import builtins
        from touyama_v3 import tv3_attacker_plant_controller as controller_module
        order=DIAGNOSTIC_PLAN_ORDER_OVERRIDES.get(c.game.battle_tick)
        if order and c.snapshot is not None and c.snapshot.round_number==TARGET_ROUND:
            def sorted_probe(rows,key=None,reverse=False):
                rows=list(rows)
                if rows and all(hasattr(a,'has_spike') for a in rows) and any(a.has_spike for a in rows):
                    ranks={slot:i for i,slot in enumerate(order)}
                    result=builtins.sorted(rows,key=lambda a:(ranks.get(a.slot,len(ranks)),key(a) if key else a.slot),reverse=reverse)
                    event=dict(tick=c.game.battle_tick,slots=[a.slot for a in result])
                    if event not in output['plan_order_overrides']:
                        output['plan_order_overrides'].append(event)
                    return result
                return builtins.sorted(rows,key=key,reverse=reverse)
            with patch.object(controller_module,'sorted',sorted_probe,create=True):
                original_prepare(c)
        else:
            original_prepare(c)
        if c.snapshot is None or c.snapshot.round_number!=TARGET_ROUND:
            return
        from frc_v1.actions import to_game_action
        names={a.slot:a.name for a in c.snapshot.allies}
        for name,(chosen,inputs,ally) in list(c.plans.items()):
            key=(c.snapshot.tick,ally.slot)
            if key not in DIAGNOSTIC_CHOICE_OVERRIDES or key in applied_choices:
                continue
            selected=DIAGNOSTIC_CHOICE_OVERRIDES[key]
            if not inputs.mask[selected]:
                raise ValueError(f'Diagnostic override is not legal: {key}, action={selected}')
            c.plans[name]=(selected,inputs,ally)
            c.actions[name]=to_game_action(c.snapshot,ally.slot,inputs.actions[selected],names)
            applied_choices.add(key)
            output['choice_overrides'].append(dict(tick=key[0],slot=key[1],original=int(chosen),selected=int(selected)))
    original_candidates=route_analysis.candidate_routes
    deadline_probe_active={'enabled':False}
    def candidates(scenario,origin,remaining_ticks,*args,**kwargs):
        rows=original_candidates(scenario,origin,remaining_ticks,*args,**kwargs)
        if not rows and RELEASE_DEADLINE_ROUTE_RESERVE and deadline_probe_active['enabled']:
            return original_candidates(scenario,origin,remaining_ticks+route_analysis.TIME_MARGIN_TICKS,*args,**kwargs)
        return rows
    output['helper_goal_changes']=[]
    original_goal=TouyamaV3AttackerPlantController.goal_for
    def goal_for(c,snapshot,ally):
        goal=original_goal(c,snapshot,ally)
        holder=next((a for a in snapshot.allies if a.alive and a.has_spike),None)
        holder_inputs=c.inputs.get(holder.name) if holder else None
        blocks_holder=holder_inputs is not None and holder_inputs.goal==ally.position
        traffic=traffic_distances(snapshot,ally,goal)[ally.position]
        static=distance_map(c.scenario.grid,goal)[ally.position]
        if (REASSIGN_BLOCKED_PLANT_HELPER and snapshot.round_number==TARGET_ROUND
                and not ally.has_spike and blocks_holder
                and c.scenario.grid[ally.position]==2
                and (traffic<0 or static>=0 and traffic-static>MAX_TRAFFIC_DETOUR_STEPS)):
            occupied={a.position for a in snapshot.allies if a.alive and a.slot!=ally.slot}
            alternatives=[]
            for p in c.scenario.sites[c.route.site]:
                if p==ally.position or p in occupied:
                    continue
                distance=traffic_distances(snapshot,ally,p)[ally.position]
                if 0<=distance<=MAX_TRAFFIC_DETOUR_STEPS:
                    alternatives.append((int(distance),p))
            if alternatives:
                replacement=min(alternatives)[1]
                c.endpoints[ally.slot]=replacement
                output['helper_goal_changes'].append(dict(tick=snapshot.tick,slot=ally.slot,
                    old=goal,new=replacement,position=ally.position))
                return replacement
        return goal
    original_encode=PlantEncoder.encode
    evidence={}
    def encode(encoder,snapshot,*args,**kwargs):
        if CONFIRMED_EMPTY_VIEW and snapshot.round_number==TARGET_ROUND:
            visible=set(snapshot.visible_cells)
            if evidence.get('tick')!=snapshot.tick:
                previous=evidence.get('visible',set())
                previous_seen=evidence.get('seen',set())
                evidence.update(tick=snapshot.tick,visible=visible,seen={s.position for s in snapshot.sightings},
                    empty=visible.intersection(previous)-previous_seen)
            snapshot=replace(snapshot,visible_cells=tuple(sorted(evidence['empty'])))
        return original_encode(encoder,snapshot,*args,**kwargs)
    original_update=AdaptiveAttackPlanner.update
    def update(planner,snapshot,observation,encoder,model,*args,**kwargs):
        if NET_PROGRESS_CLOCK and snapshot.round_number==TARGET_ROUND:
            holder=next((a for a in snapshot.allies if a.alive and a.has_spike),None)
            if holder:
                distance=min(int(f[holder.position]) for f in encoder.scenario.site_dist.values() if f[holder.position]>=0)
                if not hasattr(planner,'diagnostic_best_distance') or distance<planner.diagnostic_best_distance:
                    planner.diagnostic_best_distance=distance
                    planner.diagnostic_progress_tick=snapshot.tick
                kwargs['progress_tick_override']=planner.diagnostic_progress_tick
        if FORCE_DIAGNOSTIC_SITE is not None and snapshot.round_number==TARGET_ROUND:
            original_analyze=model.analyze
            def analyze(*a,**k):
                result=original_analyze(*a,**k)
                result['candidates']=[c for c in result['candidates'] if c['route'].site==FORCE_DIAGNOSTIC_SITE]
                return result
            with patch.object(model,'analyze',analyze):
                return original_update(planner,snapshot,observation,encoder,model,*args,**kwargs)
        return original_update(planner,snapshot,observation,encoder,model,*args,**kwargs)
    original= TouyamaV3AttackerPlantController._restrict_movement
    def restrict(c,ally,inputs,reserved):
        inputs.pre_mask_teacher=inputs.teacher
        inputs.reserved_before=tuple(sorted(reserved))
        original(c,ally,inputs,reserved)
        if RESTORE_CONFIRMED_FLASH_PROGRESS and c.snapshot.round_number==TARGET_ROUND:
            tactical=c.encoder.empty_evidence.tactical_snapshot(c.snapshot,temporary_body_cover=c.encoder.execution_version>=16)
            threats=public_threats(tactical,c.analysis_encoder.history.tracks)
            for index,kind in zip(range(8,40,8),('N','E','S','W')):
                if set(inputs.movement_debug.get(kind,()))!={'recent_edge_limit'}:
                    continue
                dr,dc=MOVE_STEPS[kind]; destination=ally.position[0]+dr,ally.position[1]+dc
                if destination not in c.route.cells or destination in reserved:
                    continue
                if inputs.traffic[ally.position]<0 or not 0<=inputs.traffic[destination]<inputs.traffic[ally.position]:
                    continue
                incoming=[t for t in threats if incoming_clear(c.scenario,tactical,ally,destination,t)]
                if incoming and all(c.encoder.flash_tracker.suppressed(c.snapshot,t.position,c.analysis_encoder.history.tracks) for t in incoming):
                    inputs.mask[index:index+8]=inputs.engine_movement_mask[index:index+8]
        if not RESTORE_UNIQUE_PROGRESS_EDGE or c.snapshot.round_number!=TARGET_ROUND:
            return
        moves=[(i,inputs.actions[i].kind) for i in range(8,40,8) if any(inputs.engine_movement_mask[i:i+8])]
        if len(moves)!=1 and not RESTORE_ALL_ROUTE_PROGRESS_EDGES:
            return
        for index,kind in moves:
            if set(inputs.movement_debug.get(kind,()))!={'recent_edge_limit'}:
                continue
            dr,dc=MOVE_STEPS[kind]; destination=ally.position[0]+dr,ally.position[1]+dc
            field=inputs.traffic
            if destination in reserved or field[ally.position]<0 or not 0<=field[destination]<field[ally.position]:
                continue
            if RESTORE_ALL_ROUTE_PROGRESS_EDGES and destination not in c.route.cells:
                continue # Do not bypass a route-corridor limit that the earlier edge branch hid.
            for i in range(index,index+8):
                if inputs.engine_movement_mask[i]:
                    inputs.mask[i]=True
    def observe(c):
        s=c.snapshot
        if s.round_number!=TARGET_ROUND:
            return
        if DEFER_INITIAL_ASSEMBLY:
            c.site_assembly.bind_changed_entry_only=False
        state=c.site_entry_state or {}
        actors=[]
        chosen_kinds={a.slot:c.plans[a.name][1].actions[c.plans[a.name][0]].kind
                      for a in s.allies if a.alive and a.name in c.plans}
        planned=planned_team_snapshot(s,chosen_kinds)
        for a in s.allies:
            if not a.alive or a.name not in c.plans:
                continue
            chosen,inputs,_=c.plans[a.name]
            with torch.no_grad():
                q=c.policy(torch.as_tensor(inputs.observation).unsqueeze(0))[0].numpy()
            ranked=sorted(np.flatnonzero(inputs.mask),key=lambda i:float(q[i]),reverse=True)[:5]
            tactical=c.encoder.empty_evidence.tactical_snapshot(s,temporary_body_cover=c.encoder.execution_version>=16) if c.encoder.execution_version>=15 else s
            threats=public_threats(tactical,c.analysis_encoder.history.tracks)
            geometry={}
            for kind in ('STAY','N','E','S','W'):
                dr,dc=MOVE_STEPS.get(kind,(0,0)); p=a.position[0]+dr,a.position[1]+dc
                if not (0<=p[0]<c.scenario.grid.shape[0] and 0<=p[1]<c.scenario.grid.shape[1]) or c.scenario.grid[p]==1:
                    continue
                incoming=[t for t in threats if incoming_clear(c.scenario,tactical,a,p,t)]
                geometry[kind]=dict(position=p,traffic=float(inputs.traffic[p]),
                    incoming=len(incoming),returnable=sum(shot_clear(c.scenario,s,p,t.position,a.slot) for t in incoming),
                    planned_incoming=sum(incoming_clear(c.scenario,planned,a,p,t) for t in threats),
                    support=[target_support(c.scenario,s,t.position,a.slot) for t in incoming])
            actors.append(dict(slot=a.slot,position=a.position,carrier=a.has_spike,hp=a.hp,reveal=a.reveal,
                observation_hash=hashlib.sha256(inputs.observation.tobytes()).hexdigest(),
                legal_mask_hash=hashlib.sha256(inputs.mask.tobytes()).hexdigest(),
                charges=a.charges,ability=a.ability_name,plant_progress=a.plant_progress,
                plant_legal=bool(inputs.mask[-2]),chosen_index=int(chosen),teacher_index=int(inputs.teacher),
                top_legal=[dict(index=int(i),kind=inputs.actions[i].kind,facing=inputs.actions[i].facing,value=float(q[i])) for i in ranked],
                goal=inputs.goal,combat=inputs.combat.reason,teacher=inputs.actions[inputs.teacher].kind,
                advice_position=inputs.combat.position,advice_facing=inputs.combat.facing,
                facing=a.facing,chosen_facing=inputs.actions[chosen].facing,
                pre_mask_teacher=inputs.actions[inputs.pre_mask_teacher].kind,
                pre_mask_teacher_facing=inputs.actions[inputs.pre_mask_teacher].facing,
                teacher_changed=inputs.pre_mask_teacher!=inputs.teacher,
                reserved_before=inputs.reserved_before,geometry=geometry,
                action=inputs.actions[chosen].kind,legal_moves={kind:any(inputs.mask[i:i+8])
                    for i,kind in zip(range(0,40,8),('STAY','N','E','S','W'))},
                movement_debug=inputs.movement_debug))
        output['frames'].append(dict(tick=s.tick,timer=s.round_timer,site=c.route.site,
            alive_allies=[dict(slot=a.slot,position=a.position,hp=a.hp,has_plan=a.name in c.plans)
                          for a in s.allies if a.alive],
            geometry_evidence=f'plant_v{c.encoder.execution_version}',
            entry_yield_events=[dict(e) for e in c.events if e['type']=='entry_yield'],
            friendly_flash=dict(records=[dict(r) for r in c.encoder.flash_tracker.records],
                features=c.encoder.flash_tracker.features(s,c.analysis_encoder.history.tracks),
                threats=[dict(enemy_id=t.enemy_id,position=t.position,
                              suppressed=c.encoder.flash_tracker.suppressed(s,t.position,c.analysis_encoder.history.tracks))
                         for t in public_threats(s,c.analysis_encoder.history.tracks)]),
            public_flashes=[dict(position=e.position,phase=e.phase) for e in s.effects if e.kind=='FLASH'],
            assembly={k:state.get(k) for k in ('active','wait','elapsed','launched','goals','ready','main','flank')},
            assembly_started=c.site_assembly.started,actors=actors,
            sightings=[dict(enemy_id=e.enemy_id,position=e.position,source=e.source) for e in s.sightings],
            tracks=[dict(enemy_id=i,position=p,age=s.tick-tick,geometrically_visible=p in s.visible_cells,
                         currently_seen=any(t.enemy_id==i for t in s.sightings))
                    for i,(p,tick,_) in c.analysis_encoder.history.tracks.items()],
            planner_debug=c.route_planner.decision_debug))
        output['frames'][-1]['progress_tick']=c.route_planner.progress_tick
        output['frames'][-1]['route_reason']=c.attack_plan.reason if c.attack_plan else None
    def completed(row):
        if row['round']<TARGET_ROUND:
            output['preceding_rounds'].append({k:row[k] for k in ('round','planted','reason','tick','alive','enemy_alive')})
        if row['round']==TARGET_ROUND:
            output['terminal']=row
            raise TargetFinished()
    def started(controller,round_number):
        deadline_probe_active['enabled']=round_number==TARGET_ROUND
        if round_number==TARGET_ROUND:
            # Hash internal RNG only for equality diagnostics; never model input.
            output['target_start_rng_hash']=hashlib.sha256(pickle.dumps((random.getstate(),
                np.random.get_state(),torch.get_rng_state().numpy().tobytes()))).hexdigest()
            if (CANDIDATE_VERSION is not None or candidate is not None) and CANDIDATE_TARGET_ONLY:
                controller.root_policy=policy
                controller.policy=policy
                controller.encoder=PlantEncoder(scenario,policy.execution_version)
                if getattr(policy,'custom_extension_contract',None) is not None:
                    from touyama_v3.tv3_model_extensions import load_extensions
                    controller.custom_pipeline=load_extensions(policy.custom_extension_contract,
                        role='attacker_plant',opponent=OPPONENT)
                    controller.reset_round()
            if FORCE_TEACHER_TARGET:
                controller.training=True
                controller.teacher_probability=1.
                controller.epsilon=0.
    with patch.object(TouyamaV3AttackerPlantController,'prepare_team_tick',prepare), \
            patch.object(TouyamaV3AttackerPlantController,'_restrict_movement',restrict), \
            patch.object(AdaptiveAttackPlanner,'update',update), patch.object(PlantEncoder,'encode',encode), \
            patch.object(TouyamaV3AttackerPlantController,'goal_for',goal_for), \
            patch.object(route_analysis,'candidate_routes',candidates):
        try:
            rollout(OPPONENT,scenario,source['analysis'],source['plant'] if CANDIDATE_TARGET_ONLY else policy,
                dict(max_round_steps=400,combat_diagnostics=COMBAT_DIAGNOSTICS),SEED,preset_name='Touyama Gaming',
                tick_callback=observe,round_callback=completed,round_start_callback=started,start_round=START_ROUND)
        except TargetFinished:
            pass
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2,default=json_value),encoding='utf8')
    print('Saved:',OUTPUT_PATH,'frames:',len(output['frames']),flush=True)


if __name__=='__main__':
    main()
