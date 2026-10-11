"""Public-contact reassessment and site-switch penalty ablations; no learning."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
BEST_DIRECTORY=HERE/'data'/'best'
LATEST_PATH=HERE/'data'/'attacker_analysis'/OPPONENT/'latest.pt'
BASELINE_PATH=HERE/'logs'/'attacker_route_risk_review'/'entry_binding_latest_changes_only.json'
OUTPUT_DIRECTORY=HERE/'logs'/'attacker_route_risk_review'
OUTPUT_PREFIX='site_switch_review'
VARIANTS=('control','fresh_contact','no_extra_site_penalty')
STOP_REGRESSED_PILOT=True
TORCH_THREADS=1

import hashlib
import json
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_learn_attacker_analysis import load_analysis
from touyama_v3.tv3_train_attacker_plant import rollout
from touyama_v3.tv3_train_attacker_analysis import summarize
from touyama_v3.tv3_learn_attacker_fire_support import public_threats
from touyama_v3.tv3_review_attacker_right import write_row,json_default


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    latest,_,state=load_analysis(LATEST_PATH,scenario,OPPONENT)
    baseline=json.loads(BASELINE_PATH.read_text(encoding='utf-8'))
    digest=hashlib.sha256(LATEST_PATH.read_bytes()).hexdigest()
    if (baseline['source_hashes']!=source['hashes'] or baseline['analysis_hash']!=digest
        or state['config'].get('collection_source_hashes')!=source['hashes']):
        raise ValueError('Reference/executor mismatch')
    seeds=sorted({r['seed'] for r in baseline['rounds']})
    config=load_checkpoint(source['paths']['plant'])['config']
    output=dict(scope='experimental_switch_policy_not_adopted',source_hashes=source['hashes'],
        analysis_hash=digest,optimizer_updates=0,variants={})
    for variant in VARIANTS:
        records=[]
        rejected=False
        trace=OUTPUT_DIRECTORY/f'{OUTPUT_PREFIX}_{variant}.jsonl'
        with trace.open('w',encoding='utf-8') as stream:
            for seed in seeds[:1] if variant=='control' else seeds:
                def starting(controller,round_number):
                    controller.site_assembly.follow_route_entry=True
                    controller.site_assembly.bind_changed_entry_only=True
                    controller.route_planner.reassess_new_contacts=variant=='fresh_contact'
                    if variant=='no_extra_site_penalty':
                        controller.route_planner.site_switch_margin=None
                        controller.route_planner.switch_travel_cost=0.
                def observe(controller):
                    snap=controller.snapshot
                    if snap.phase=='setup' or controller.attack_plan is None:
                        return
                    threats=public_threats(snap,controller.analysis_encoder.history.tracks)
                    counts={side:sum(scenario.site_of(t.position)==side for t in threats) for side in ('L','R')}
                    debug=controller.route_planner.decision_debug
                    write_row(stream,dict(seed=seed,round=snap.round_number,tick=snap.tick,site=controller.route.site,
                        known_half_counts=counts,public_sightings=[dict(id=s.enemy_id,position=s.position) for s in snap.sightings],
                        review=debug if debug and debug['tick']==snap.tick else None,
                        carrier=[dict(position=a.position,goal=x.goal,action=x.actions[i].kind)
                                 for i,x,a in controller.plans.values() if a.has_spike]))
                rows,_=rollout(OPPONENT,scenario,latest,source['plant'],config,seed,
                    preset_name='Touyama Gaming',training=False,round_start_callback=starting,tick_callback=observe)
                records.extend(dict(seed=seed,**r) for r in rows)
                reference=[r for r in baseline['rounds'] if r['seed']==seed]
                if variant=='control':
                    fields=('round','planted','tick','alive','remaining_hp','reason','site')
                    if len(rows)!=len(reference) or any(any(a[k]!=b[k] for k in fields) for a,b in zip(rows,reference)):
                        raise RuntimeError('Control does not reproduce baseline')
                current,old=summarize(rows),summarize(reference)
                rejected=(variant!='control' and STOP_REGRESSED_PILOT and seed==seeds[0]
                          and (current['plants']<old['plants'] or current['viable_plants']<old['viable_plants']))
                output['variants'][variant]=dict(summary=summarize(records),rounds=records,
                    rejected_pilot=rejected,completed_seeds=len(records)//12,trace=str(trace))
                (OUTPUT_DIRECTORY/f'{OUTPUT_PREFIX}_summary.json').write_text(
                    json.dumps(output,default=json_default,ensure_ascii=False,indent=2),encoding='utf-8')
                print(variant,seed,'plants',current['plants'],'viable',current['viable_plants'],
                      'stop_pilot',rejected,flush=True)
                if rejected:
                    break
    print('Saved:',OUTPUT_DIRECTORY/f'{OUTPUT_PREFIX}_summary.json',flush=True)


if __name__=='__main__':
    main()
