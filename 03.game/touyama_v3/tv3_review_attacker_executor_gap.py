"""Reproduce saved analysis evaluation with rule and adopted plant executors; no learning."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
BEST_DIRECTORY=HERE/'data'/'best'
OUTPUT_PATH=HERE/'logs'/'attacker_route_risk_review'/'executor_gap.json'
CASE_SEEDS=None  # None uses the adopted analysis checkpoint evaluation seeds.
TORCH_THREADS=1

import json
import numpy as np
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_collect_attacker_analysis import play_block
from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder
from touyama_v3.tv3_train_attacker_analysis import summarize


def first_contact_predictions(model,encoder,rounds,samples):
    rows=[]
    terminal={r['round']:r for r in rounds}
    columns=[encoder.fields.index(f'enemy_{i}_current') for i in range(5)]
    with torch.inference_mode():
        for sample in samples:
            contact=np.flatnonzero((sample['observations'][:,columns]>.5).any(axis=1)
                                    & (sample['entry_outside'][:,0]>=0))
            if not len(contact):
                continue
            i=int(contact[0])
            _,raw=model(torch.tensor(sample['observations'][i:i+1].astype(np.float32)),
                        torch.tensor(sample['routes'][i:i+1].astype(np.float32)))
            p=float(raw.sigmoid()[0,0])
            outcome=terminal[sample['round']]
            rows.append(dict(round=sample['round'],tick=int(sample['ticks'][i]),
                outside=sample['entry_outside'][i].tolist(),site='R' if sample['selected_sites'][i] else 'L',
                prediction=p,planted=outcome['planted'],actual_site=outcome['site']))
    return rows


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    state=load_checkpoint(source['paths']['analysis'])
    encoder=AttackerEncoder(scenario,source['analysis'].execution_version)
    seeds=CASE_SEEDS if CASE_SEEDS is not None else state['evaluation']['seeds']
    output=dict(scope='executor_comparison_same_seeds_not_same_intermediate_state_or_fixed_entry_causality',
        source_hashes=source['hashes'],scenario=scenario.signature,seeds=seeds,
        saved_rule_evaluation=state['evaluation']['selected'],executors={},optimizer_updates=0)
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    for kind in ('rule','fixed_plant'):
        config={**state['config'],'collection_executor':kind}
        all_rounds=[]; predictions=[]
        for seed in seeds:
            rounds,samples=play_block(OPPONENT,scenario,source['analysis'],config,seed,0.,
                preset_name='Touyama Gaming',plant_policy=source['plant'] if kind=='fixed_plant' else None)
            all_rounds.extend(dict(seed=seed,**r) for r in rounds)
            predictions.extend(dict(seed=seed,**r) for r in first_contact_predictions(source['analysis'],encoder,rounds,samples))
            output['executors'][kind]=dict(summary=summarize(all_rounds),rounds=all_rounds,predictions=predictions,
                                          completed_seeds=len(all_rounds)//12)
            OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
            print(kind,seed,'plants',sum(r['planted'] for r in rounds),'/12',flush=True)
    saved=output['saved_rule_evaluation']; reproduced=output['executors']['rule']['summary']
    output['saved_rule_reproduction']={k:reproduced[k]==saved[k] for k in (
        'rounds','plants','viable_plants','mean_alive','mean_damage','mean_ability_uses','mean_plant_ticks','timeouts')}
    OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved rule reproduction:',output['saved_rule_reproduction'],flush=True)
    print('Saved:',OUTPUT_PATH,flush=True)


if __name__=='__main__':
    main()
