"""Reproduce the saved high-confidence group and compare evaluation executors; no training."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
ANALYSIS_PATH=HERE/'data'/'best'/OPPONENT/'attacker_analysis_best.pt'
PINNED_DIRECTORY=HERE/'data'/'attacker_analysis'/'collection_sources'
OUTPUT_PATH=HERE/'logs'/'attacker_route_risk_review'/'analysis_10of11_reproduction.json'
COMPARE_PLANT_BLOCKS=1  # Same-model executor comparison; stop if any discrepancy appears.
TORCH_THREADS=1

import hashlib
import json
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_learn_attacker_analysis import load_analysis
from touyama_v3.tv3_collect_attacker_analysis import play_block
from touyama_v3.tv3_train_attacker_plant import rollout
from touyama_v3.tv3_train_attacker_analysis import entry_prediction_metrics


def first_contact(row):
    return next((d for d in row['route_decisions']
        if d.get('public_contact_count',0)>0 and d.get('predicted_plant_probability') is not None
        and d.get('entry_boundary',{}).get('outside',[-1])[0]>=0),None)


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    model,_,saved=load_analysis(ANALYSIS_PATH,scenario,OPPONENT)
    source=load_sources(scenario,(OPPONENT,),PINNED_DIRECTORY,PINNED_DIRECTORY)[OPPONENT]
    if source['hashes']!=saved['config']['collection_source_hashes']:
        raise ValueError('Pinned source differs from the saved analysis evaluation')
    config={**saved['config'],'combat_diagnostics':False}
    expected=saved['evaluation']['selected_entry_prediction']['groups']['R:11,40:p>=0.9']
    output=dict(optimizer_updates=0,analysis_hash=hashlib.sha256(ANALYSIS_PATH.read_bytes()).hexdigest(),
        analysis_set=saved['completed_sets'],plant_version=source['plant'].execution_version,
        source_hashes=source['hashes'],expected=expected,blocks=[],high_confidence_rounds=[])
    all_rounds=[]
    compared=0
    for seed in saved['evaluation']['seeds']:
        for start in saved['config']['evaluation_start_rounds']:
            traces=[]
            def observe(controller):
                snap=controller.snapshot
                traces.append((snap.round_number,snap.tick,
                    hashlib.sha256(repr(snap).encode('utf8')).hexdigest(),
                    hashlib.sha256(json.dumps(controller.actions,sort_keys=True,default=str).encode('utf8')).hexdigest()))
            rounds,_=play_block(OPPONENT,scenario,model,config,seed,0.,
                preset_name='Touyama Gaming',plant_policy=source['plant'],start_round=start,
                tick_callback=observe if compared<COMPARE_PLANT_BLOCKS else None)
            block=dict(seed=seed,start_round=start,plants=sum(r['planted'] for r in rounds))
            if compared<COMPARE_PLANT_BLOCKS:
                first_traces=list(traces); traces.clear()
                other,_=rollout(OPPONENT,scenario,model,source['plant'],config,seed,
                    preset_name='Touyama Gaming',start_round=start,tick_callback=observe)
                keys=('round','planted','site','tick','alive','remaining_hp','actual_paths')
                block['same_model_plant_evaluator']=dict(
                    public_states_and_actions_equal=first_traces==traces,
                    compared_ticks=len(first_traces),
                    outcomes_equal=[{k:r[k] for k in keys} for r in rounds]==[{k:r[k] for k in keys} for r in other])
                block['record_differences']=[dict(round=r['round'],fields=[k for k in (*keys,'route_decisions')
                    if r[k]!=o[k]]) for r,o in zip(rounds,other) if any(r[k]!=o[k] for k in (*keys,'route_decisions'))]
                compared+=1
                if not all(block['same_model_plant_evaluator'][k] for k in ('public_states_and_actions_equal','outcomes_equal')):
                    output['blocks'].append(block)
                    OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
                    raise RuntimeError('Executor discrepancy: stopped before further blocks')
            output['blocks'].append(block)
            for row in rounds:
                d=first_contact(row)
                if d and d['site']=='R' and tuple(d['entry_boundary']['outside'])==(11,40) and d['predicted_plant_probability']>=.9:
                    output['high_confidence_rounds'].append(dict(seed=seed,round=row['round'],
                        contact_tick=d['tick'],prediction=d['predicted_plant_probability'],
                        planted=row['planted'],actual_site=row['site'],terminal_tick=row['tick'],
                        alive=row['alive'],remaining_hp=row['remaining_hp'],reason=row['reason'],
                        carrier_site_entries=row['carrier_site_entries'],actual_paths=row['actual_paths']))
            all_rounds.extend(rounds)
            output['reproduced']=entry_prediction_metrics(all_rounds)
            OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
            OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
            print(json.dumps(block),flush=True)
    actual=output['reproduced']['groups']['R:11,40:p>=0.9']
    output['saved_group_reproduced']=all(actual[k]==expected[k] for k in ('rounds','plants','mean_prediction','brier'))
    OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    print('High confidence:',json.dumps(actual),'exact saved match:',output['saved_group_reproduced'],flush=True)


if __name__=='__main__':
    main()
