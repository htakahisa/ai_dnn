"""Controlled right routes and shot-time labels. No optimization/checkpoint writes."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OPPONENT = 'frc_v1'
BEST_DIRECTORY = HERE / 'data' / 'best'
OUTPUT_DIRECTORY = HERE / 'logs' / 'attacker_approach_review'
EVALUATION_SEEDS = (804000042, 804000142, 804000242)
TARGET_ROUND = 6
ROUTE_INDEXES = (None, 0, 1, 2, 3, 4, 5)  # None: natural planner; others: fixed right geometry.
POLICY_MODE = 'learned'  # learned / teacher_v10 / teacher_v11
OUTPUT_PREFIX = 'approach'
PRESET = 'Touyama Gaming'
TORCH_THREADS = 1

import argparse
from collections import Counter
import hashlib
import json
import random
import numpy as np
import torch
from grid_lines import line_cells
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_analysis import candidate_routes
from touyama_v3.tv3_attacker_route_planner import AttackPlan, WAYPOINT_STEPS, ENTRY_STEPS
from touyama_v3.tv3_learn_attacker_plant import PlantDQN
from touyama_v3.tv3_learn_attacker_fire_support import public_threats
from touyama_v3.tv3_attacker_combat import RECENT_CONTACT_TICKS
from touyama_v3.tv3_train_attacker_plant import rollout
from touyama_v3.tv3_review_attacker_right import write_row, json_default


def memory_status(snapshot, tracks):
    current = {s.enemy_id for s in snapshot.sightings}
    alive = {e.enemy_id for e in snapshot.enemies if e.alive}
    valid = {s.enemy_id for s in public_threats(snapshot, tracks)}
    return [dict(enemy_id=i,position=pos,age=snapshot.tick-tick,
        state='current' if i in current else 'dead' if i not in alive else
              'expired' if snapshot.tick-tick > RECENT_CONTACT_TICKS else
              'retained' if i in valid else 'visible_without_current_sighting')
            for i,(pos,tick,_) in tracks.items()]


class FixedRouteProbe:
    """Diagnostic geometry commitment. Local actions remain policy/teacher choices."""
    def __init__(self, route):
        self.route = route
        self.events = []
        self.flank_entries = {}
        self.progress_tick = 0

    def update(self, snapshot, observation, encoder, model, old=None, mode=None, **kwargs):
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        origin = holder.position if holder else snapshot.spike_dropped
        cells = self.route.cells
        index = cells.index(origin) if origin in cells else 0
        phase = 'entry' if len(cells)-index <= ENTRY_STEPS else 'approach'
        changed = old is None
        if changed:
            self.events.append(dict(tick=snapshot.tick,site='R',key=self.route.key,
                                    branches=self.route.branches,reason='diagnostic_fixed_geometry'))
        return AttackPlan(self.route,mode or 'supported',phase,changed,'diagnostic_fixed_geometry',
                          cells[min(index+WAYPOINT_STEPS,len(cells)-1)],{})


class RoundComplete(Exception):
    def __init__(self, record):
        self.record = record


class ApproachAudit:
    """Public decisions first; private shot-scene facts only after actions freeze."""
    def __init__(self, stream, seed, index, scenario, policy, mode, *, shot_labels=True):
        self.stream,self.seed,self.index,self.scenario,self.policy,self.mode = stream,seed,index,scenario,policy,mode
        self.installed = False
        self.intervened = False
        self.public = None
        self.tracks = {}
        self.counts = Counter()
        self.initial_hash = None
        self.selected = None
        self.shot_labels = shot_labels

    def starting(self, controller, round_number):
        if round_number == TARGET_ROUND and self.mode == 'teacher_exposure':
            from touyama_v3.tv3_review_attacker_exposure_teacher import install_probe
            install_probe(controller)

    def observe(self, controller):
        game = controller.game
        snap = controller.snapshot
        if not self.installed:
            original = game._resolve_all_shots
            def resolve(*args,**kwargs):
                if game.current_round != TARGET_ROUND or game.is_planted or self.public is None:
                    return original(*args,**kwargs)
                # After movement/abilities, before shooting. These labels are never
                # passed into the planner, observation, teacher or action selector.
                smoke = set(game._smoke_cells())
                scene = {str(c.name):dict(position=tuple(c.pos),facing=c.facing,
                    blind=c.blind_remaining,recon=c.reveal_remaining,
                    through_smoke=bool(getattr(c,'sees_through_smoke',False)),
                    los_revealed=bool(c.los_revealed)) for c in game.chars}
                result = original(*args,**kwargs)
                seen = {s.enemy_id for s in self.public.sightings}
                enemy_ids = {e.name:e.enemy_id for e in self.public.enemies}
                for shot in game.last_shots:
                    shooter,target = shot['shooter'],shot['target']
                    if shooter.team != 'D' or target.team != 'A' or str(shooter.name) not in scene or str(target.name) not in scene:
                        continue
                    s,t = scene[str(shooter.name)],scene[str(target.name)]
                    source_id = enemy_ids.get(str(getattr(shooter,'base_name',shooter.name)))
                    smoke_line = bool(smoke.intersection(line_cells(s['position'],t['position'])))
                    unseen = source_id not in seen
                    cause = ('observed' if not unseen else 'smoke_bypass_target_recon' if smoke_line and t['recon'] else
                             'smoke_bypass_shooter_trait' if smoke_line and s['through_smoke'] else
                             'blind_observers' if all(a.blind for a in self.public.allies if a.alive) else
                             'first_contact_after_movement' if not s['los_revealed'] else 'other_unseen')
                    self.counts['incoming'] += 1
                    self.counts[cause] += 1
                    self.counts['damage_'+cause] += shot['damage']
                    write_row(self.stream,dict(event='shot_label',seed=self.seed,route_index=self.index,
                        round=TARGET_ROUND,tick=self.public.tick,classification=cause,
                        source_id=source_id,source_seen_before_action=not unseen,
                        smoke_line=smoke_line,shooter=s,target=t,damage=shot['damage'],hit=shot['hit'],
                        note='Private post-decision diagnostic label, not policy input'))
                return result
            if self.shot_labels:
                game._resolve_all_shots = resolve
            self.installed = True
        if snap.round_number != TARGET_ROUND:
            return
        if not self.intervened:
            # Compare physical initial states and RNG, not just the requested route.
            label = dict(chars=[(str(c.name),c.team,tuple(c.pos),c.hp,c.facing,c.blind_remaining,c.reveal_remaining)
                                for c in game.chars],timer=game.round_timer,
                         python_rng=repr(random.getstate()),numpy_rng=repr(np.random.get_state()))
            self.initial_hash = hashlib.sha256(json.dumps(label,default=json_default).encode()).hexdigest()
            holder = next(a for a in snap.allies if a.alive and a.has_spike)
            routes = [r for r in candidate_routes(self.scenario,holder.position,snap.round_timer) if r.site=='R']
            if self.index is not None:
                if self.index >= len(routes):
                    raise ValueError(f'Only {len(routes)} right routes available')
                self.selected = routes[self.index]
                controller.route_planner = FixedRouteProbe(self.selected)
                controller.route = controller.route_mode = None
                controller.cache = None
                controller.prepare_team_tick()
                snap = controller.snapshot
            self.intervened = True
            write_row(self.stream,dict(event='initial',seed=self.seed,route_index=self.index,
                initial_state_hash=self.initial_hash,candidates=[dict(key=r.key,branches=r.branches,cells=r.cells) for r in routes],
                selected_key=self.selected.key if self.selected else None,
                scope='diagnostic_geometry_commitment_not_normal_match' if self.index is not None else 'natural_planner'))
        self.public = snap
        self.tracks = dict(controller.analysis_encoder.history.tracks)
        actors = []
        for name,(selected,inputs,ally) in controller.plans.items():
            actors.append(dict(slot=ally.slot,position=ally.position,hp=ally.hp,reveal=ally.reveal,blind=ally.blind,
                goal=inputs.goal,carrier=ally.has_spike,action=inputs.actions[selected].kind,
                teacher=inputs.actions[inputs.teacher].kind,combat=inputs.combat.reason,
                contacts=inputs.combat.contacts,future_contacts=inputs.combat.future_contacts,
                supporters=inputs.combat.supporters,wait_ticks=inputs.combat.wait_ticks,
                exposure_probe=getattr(inputs,'exposure_debug',None),
                excluded=getattr(inputs,'movement_debug',{})))
        write_row(self.stream,dict(event='decision',seed=self.seed,route_index=self.index,round=TARGET_ROUND,
            tick=snap.tick,site=controller.route.site,actors=actors,memory=memory_status(snap,self.tracks),
            smoke_cells=snap.smoke_cells,effects=[dict(kind=e.kind,phase=e.phase,position=e.position) for e in snap.effects],
            planner_progress_tick=controller.route_planner.progress_tick))


def main():
    global TARGET_ROUND
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',nargs='+',type=int,default=EVALUATION_SEEDS)
    parser.add_argument('--routes',nargs='+',default=['natural' if i is None else str(i) for i in ROUTE_INDEXES])
    parser.add_argument('--mode',choices=('learned','teacher_v10','teacher_v11','teacher_exposure'),default=POLICY_MODE)
    parser.add_argument('--prefix',default=OUTPUT_PREFIX)
    parser.add_argument('--round',type=int,default=TARGET_ROUND)
    args=parser.parse_args()
    if not 1 <= args.round <= 12:
        raise ValueError('round must be in 1..12')
    TARGET_ROUND=args.round
    if Path(args.prefix).name != args.prefix:
        raise ValueError('prefix must be a filename component')
    indexes=[None if v=='natural' else int(v) for v in args.routes]
    if args.mode=='teacher_exposure' and indexes != [None]:
        raise ValueError('Experimental teacher uses natural route selection; specify --routes natural')
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    config=load_checkpoint(source['paths']['plant'])['config']
    policy=PlantDQN(11).eval().requires_grad_(False) if args.mode in ('teacher_v11','teacher_exposure') else source['plant']
    OUTPUT_DIRECTORY.mkdir(parents=True,exist_ok=True)
    results=[]
    from run_game import FRC_V1_ATTACKER_CHECKPOINT,FRC_V1_DEFENDER_CHECKPOINT
    opponent_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (FRC_V1_ATTACKER_CHECKPOINT,FRC_V1_DEFENDER_CHECKPOINT)}
    initial_hashes={}
    for seed in args.seeds:
        for index in indexes:
            trace=OUTPUT_DIRECTORY/f'{args.prefix}_{args.mode}_{seed}_{index}_trace.jsonl'
            with trace.open('w',encoding='utf-8') as stream:
                audit=ApproachAudit(stream,seed,index,scenario,policy,args.mode)
                def completed(record):
                    if record['round']==TARGET_ROUND:
                        raise RoundComplete(record)
                try:
                    rollout(OPPONENT,scenario,source['analysis'],policy,config,seed,preset_name=PRESET,
                        training=args.mode!='learned',teacher_probability=0. if args.mode=='learned' else 1.,
                        tick_callback=audit.observe,round_callback=completed,round_start_callback=audit.starting)
                except RoundComplete as done:
                    if audit.initial_hash is None:
                        raise RuntimeError('Round ended before diagnostic initialization')
                    reference=initial_hashes.setdefault(seed,audit.initial_hash)
                    if reference!=audit.initial_hash:
                        raise RuntimeError('Pre-intervention state/RNG differ across route probes')
                    results.append(dict(seed=seed,route_index=index,initial_state_hash=audit.initial_hash,
                        requested_branches=audit.selected.branches if audit.selected else None,
                        trace=str(trace),shot_labels=dict(audit.counts),record=done.record))
                    print(seed,index,done.record['reason'],done.record['alive'],dict(audit.counts),flush=True)
    output=OUTPUT_DIRECTORY/f'{args.prefix}_{args.mode}_summary.json'
    if opponent_hashes != {str(Path(path)):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in opponent_hashes}:
        raise RuntimeError('Opponent checkpoints changed during evaluation')
    output.write_text(json.dumps(dict(source_hashes=source['hashes'],opponent_hashes=opponent_hashes,scenario=scenario.signature,
        mode=args.mode,scope='controlled_route_diagnostic_not_learned_win_rate',target_round=TARGET_ROUND,
        teacher_probability=0. if args.mode=='learned' else 1.,optimizer_updates=0,
        intervention_timing='before_first_plan' if args.mode=='teacher_exposure' else 'after_first_plan' if any(i is not None for i in indexes) else 'none',
        exposure_probe_hash=hashlib.sha256((HERE/'tv3_review_attacker_exposure_teacher.py').read_bytes()).hexdigest()
            if args.mode=='teacher_exposure' else None,
        results=results),default=json_default,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',output)


if __name__=='__main__':
    main()
