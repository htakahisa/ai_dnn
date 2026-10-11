"""Same-state analysis replacement diagnostic with frozen plant; no learning/publication."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
BEST_DIRECTORY=HERE/'data'/'best'
LATEST_PATH=HERE/'data'/'attacker_analysis'/OPPONENT/'latest.pt'
CASE_SEED=804000042
CASE_ROUND=6
INTERVENTION_TICK=33
OUTPUT_DIRECTORY=HERE/'logs'/'attacker_route_risk_review'
OUTPUT_PREFIX='analysis_swap'
TORCH_THREADS=1
BIND_CHANGED_ENTRY_ONLY=True

import json
import hashlib
import random
import numpy as np
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_learn_attacker_analysis import load_analysis,AttackerEncoder
from touyama_v3.tv3_train_attacker_plant import rollout
from touyama_v3.tv3_review_attacker_right import write_row,json_default


class Complete(Exception):
    pass


class SwapProbe:
    def __init__(self,controller,replacement,reset_entry,stream):
        self.controller=controller
        self.base=controller.route_planner
        self.replacement=replacement
        self.reset_entry=reset_entry
        self.stream=stream
        self.initial_hash=None

    def __getattr__(self,name):
        return getattr(self.base,name)

    @property
    def progress_tick(self):
        return self.base.progress_tick

    @progress_tick.setter
    def progress_tick(self,value):
        self.base.progress_tick=value

    def update(self,snapshot,observation,encoder,model,*args,**kwargs):
        if snapshot.tick>=INTERVENTION_TICK and self.initial_hash is None:
            state=dict(snapshot=repr(snapshot),python_rng=repr(random.getstate()),numpy_rng=repr(np.random.get_state()),
                controller_rng=self.controller.rng.bit_generator.state,
                physical_labels=[(str(c.name),c.team,tuple(c.pos),c.hp,c.facing,c.blind_remaining,c.reveal_remaining)
                                 for c in self.controller.game.chars])
            self.initial_hash=hashlib.sha256(json.dumps(state).encode()).hexdigest()
            if self.replacement is not None:
                self.controller.analysis=self.replacement
                model=self.replacement
            if self.reset_entry:
                self.controller.site_assembly.follow_route_entry=True
                self.controller.site_assembly.bind_changed_entry_only=BIND_CHANGED_ENTRY_ONLY
            write_row(self.stream,dict(event='intervention',tick=snapshot.tick,
                swapped=self.replacement is not None,follow_route_entry=self.reset_entry))
        return self.base.update(snapshot,observation,encoder,model,*args,**kwargs)


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    latest,encoder,state=load_analysis(LATEST_PATH,scenario,OPPONENT)
    original=AttackerEncoder(scenario,source['analysis'].execution_version)
    if encoder.schema()!=original.schema() or state['config'].get('collection_source_hashes')!=source['hashes']:
        raise ValueError('Analysis swap diagnostic requires matching contracts and the recorded pinned executor')
    config=load_checkpoint(source['paths']['plant'])['config']
    OUTPUT_DIRECTORY.mkdir(parents=True,exist_ok=True)
    prefix=OUTPUT_PREFIX+('_changes_only' if BIND_CHANGED_ENTRY_ONLY else '_full')
    results=[]
    for variant in ('control','analysis_swap','analysis_swap_and_entry'):
        probe=None
        trace=OUTPUT_DIRECTORY/f'{prefix}_{variant}.jsonl'
        with trace.open('w',encoding='utf-8') as stream:
            def starting(controller,round_number):
                nonlocal probe
                if round_number==CASE_ROUND:
                    probe=SwapProbe(controller,latest if variant!='control' else None,
                                    variant=='analysis_swap_and_entry',stream)
                    controller.route_planner=probe
            def observe(controller):
                snap=controller.snapshot
                if snap.round_number==CASE_ROUND and snap.phase!='setup' and controller.attack_plan is not None:
                    write_row(stream,dict(event='decision',tick=snap.tick,site=controller.route.site,
                        branches=controller.route.branches,reason=controller.attack_plan.reason,
                        actors=[dict(slot=a.slot,carrier=a.has_spike,position=a.position,hp=a.hp,
                            goal=x.goal,action=x.actions[i].kind) for i,x,a in controller.plans.values()],
                        planner=controller.route_planner.events[-1] if controller.route_planner.events else None))
            def completed(record):
                if record['round']==CASE_ROUND:
                    results.append(dict(variant=variant,record=record,initial_state_hash=probe.initial_hash,trace=str(trace)))
                    raise Complete()
            try:
                rollout(OPPONENT,scenario,source['analysis'],source['plant'],config,CASE_SEED,
                    preset_name='Touyama Gaming',training=False,round_start_callback=starting,
                    tick_callback=observe,round_callback=completed)
            except Complete:
                pass
    if len({r['initial_state_hash'] for r in results})!=1 or any(r['initial_state_hash'] is None for r in results):
        raise RuntimeError('Pre-intervention physical/public state or RNG mismatch')
    output=dict(scope='diagnostic_model_swap_not_production_dependency_change',seed=CASE_SEED,round=CASE_ROUND,
        bind_changed_entry_only=BIND_CHANGED_ENTRY_ONLY,
        source_hashes=source['hashes'],replacement_hash=hashlib.sha256(LATEST_PATH.read_bytes()).hexdigest(),
        optimizer_updates=0,results=results)
    path=OUTPUT_DIRECTORY/f'{prefix}_summary.json'
    path.write_text(json.dumps(output,default=json_default,ensure_ascii=False,indent=2),encoding='utf-8')
    print([(r['variant'],r['record']['reason'],r['record']['alive']) for r in results])
    print('Saved:',path)


if __name__=='__main__':
    main()
