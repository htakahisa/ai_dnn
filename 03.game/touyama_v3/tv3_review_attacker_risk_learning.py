"""Read-only audit of saved route labels and adopted-model risk sensitivity."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
DATA_DIRECTORY=HERE/'data'/'attacker_analysis'/OPPONENT
ROUND_LOG=HERE/'logs'/'attacker_analysis'/OPPONENT/f'{OPPONENT}_rounds.jsonl'
BEST_DIRECTORY=HERE/'data'/'best'
OUTPUT_PATH=HERE/'logs'/'attacker_route_risk_review'/'risk_learning_audit.json'
EXPOSURE_THRESHOLD=.05
EXPOSURE_ABLATION_LEVEL=.2  # Within the recorded feature range; not a coherent alternate world.
TORCH_THREADS=1
RUN_FIXED_COLLECTION=False  # Optional read-only twelve-round simulation, no optimization.
FIXED_COLLECTION_SEED=804000042
FIXED_COLLECTION_CONFIG=dict(max_round_steps=20000,collection_executor='fixed_plant',combat_diagnostics=False)
FIXED_COLLECTION_OUTPUT=HERE/'logs'/'attacker_route_risk_review'/'risk_entry_credit_frozen.json'

import json
import argparse
import numpy as np
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder


def entry_credit(rows,scenario):
    counts=dict(planted_rounds=0,same_site_earlier_different_entry_decisions=0,
                planted_rounds_with_same_site_entry_change=0,
                earlier_c_decisions_before_other_right_entry=0,
                ignored_non_boundary_entries=0)
    examples=[]
    for row in rows:
        if not row.get('planted'):
            continue
        counts['planted_rounds']+=1
        decisions=row.get('route_decisions',[])
        selected=[]
        for d in decisions:
            if d.get('site')!=row['site'] or not d.get('entry'):
                continue
            p=tuple(d['entry'])
            if scenario.grid[p]==2 or not any(q in scenario.sites[row['site']] for q in scenario.neighbors(p)):
                counts['ignored_non_boundary_entries']+=1
            else:
                selected.append(d)
        if not selected:
            continue
        last=selected[-1]['entry']
        earlier=[d for d in selected if d['entry']!=last]
        counts['same_site_earlier_different_entry_decisions']+=len(earlier)
        counts['planted_rounds_with_same_site_entry_change']+=bool(earlier)
        if row['site']=='R' and tuple(last)!=(11,40):
            counts['earlier_c_decisions_before_other_right_entry']+=sum(tuple(d['entry'])==(11,40) for d in earlier)
        if earlier:
            examples.append(dict(set=row.get('set'),round=row['round'],site=row['site'],
                final_selected_entry=last,earlier_selected_entries=sorted({tuple(d['entry']) for d in earlier}),
                earlier_ticks=[d['tick'] for d in earlier],damage=row.get('damage'),alive=row.get('alive')))
    return dict(**counts,examples=examples,
        caveat='Final selected entry is a planning target, not proof of the carrier crossing. '
               'Success after adaptive continuation is not a fixed-entry success label.')


def summarize(prediction,targets):
    if not len(targets):
        return dict(frames=0)
    return dict(frames=len(targets),label_success_rate=float(targets[:,0].mean()),
                predicted_success_mean=float(prediction[:,0].mean()),
                predicted_damage_mean=float(prediction[:,1].mean()*500),
                label_damage_mean=float(targets[:,1].mean()*500))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect',action='store_true',default=RUN_FIXED_COLLECTION)
    args=parser.parse_args()
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    model=source['analysis']
    encoder=AttackerEncoder(scenario,model.execution_version)
    batches=[]
    incompatible=[]
    for path in sorted(DATA_DIRECTORY.glob('set_*.npz')):
        with np.load(path,allow_pickle=False) as data:
            for key in data.files:
                if not key.endswith('_observations'):
                    continue
                prefix=key[:-len('_observations')]
                x,z,y=(data[prefix+'_'+name].astype(np.float32) for name in ('observations','routes','outcomes'))
                if x.shape[1]!=len(encoder.fields) or z.shape[1]!=len(encoder.route_fields):
                    incompatible.append(dict(path=str(path),round=prefix,observation_size=x.shape[1],route_size=z.shape[1]))
                    continue
                batches.append((x,z,y))
    if not batches:
        raise ValueError('No compatible recorded inputs; do not relabel incompatible data')
    x,z,y=(np.concatenate([b[i] for b in batches]) for i in range(3))
    exposure_index=encoder.route_fields.index('known_exposure')
    with torch.inference_mode():
        def predict(inputs):
            return model(torch.tensor(x),torch.tensor(inputs))[1].sigmoid().numpy()
        actual=predict(z)
        low=z.copy(); low[:,exposure_index]=0.
        high=z.copy(); high[:,exposure_index]=EXPOSURE_ABLATION_LEVEL
        low_prediction,high_prediction=predict(low),predict(high)
    masks=dict(all=np.ones(len(y),bool),failed=y[:,0]<.5,
               known_exposure=z[:,exposure_index]>=EXPOSURE_THRESHOLD,
               failed_known_exposure=(y[:,0]<.5)&(z[:,exposure_index]>=EXPOSURE_THRESHOLD))
    logged=[json.loads(line) for line in ROUND_LOG.read_text(encoding='utf-8').splitlines() if line.strip()]
    result=dict(scope='recorded_data_and_feature_ablation_not_retraining_or_causal_route_success',
        caveat='These current files are not proven to be the adopted model training set. '
               'Frame-weighted summaries have correlated frames and are not round win rates. '
               'Exposure replacement alone is a feature ablation, not a coherent alternate world.',
        source_hashes=source['hashes'],scenario=scenario.signature,analysis_version=model.execution_version,
        files=[str(p) for p in sorted(DATA_DIRECTORY.glob('set_*.npz'))],incompatible=incompatible,
        groups={name:summarize(actual[mask],y[mask]) for name,mask in masks.items()},
        exposure_ablation=dict(level=EXPOSURE_ABLATION_LEVEL,zero_success_mean=float(low_prediction[:,0].mean()),
            increased_success_mean=float(high_prediction[:,0].mean()),
            mean_success_delta=float((high_prediction[:,0]-low_prediction[:,0]).mean()),
            mean_damage_delta=float((high_prediction[:,1]-low_prediction[:,1]).mean()*500),
            exposure_range=[float(z[:,exposure_index].min()),float(z[:,exposure_index].max())]),
        adaptive_entry_credit=entry_credit(logged,scenario),optimizer_updates=0)
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('groups','exposure_ablation')},indent=2))
    print('Entry-change planted rounds:',result['adaptive_entry_credit']['planted_rounds_with_same_site_entry_change'])
    print('Saved:',OUTPUT_PATH)
    if args.collect:
        from touyama_v3.tv3_collect_attacker_analysis import play_block
        rounds,samples=play_block(OPPONENT,scenario,source['analysis'],FIXED_COLLECTION_CONFIG,
            FIXED_COLLECTION_SEED,0.,preset_name='Touyama Gaming',plant_policy=source['plant'])
        counts=dict(same_site_positive_entry_frames=0,positive_frames_without_matching_future_carrier_entry=0)
        cases=[]
        sample_by_round={s['round']:s for s in samples}
        for row in rounds:
            sample=sample_by_round.get(row['round'])
            if sample is None:
                continue
            for i,(outside,inside,tick) in enumerate(zip(sample['entry_outside'],sample['entry_inside'],sample['ticks'])):
                if not row['planted'] or outside[0]<0 or ('R' if sample['selected_sites'][i] else 'L')!=row['site']:
                    continue
                counts['same_site_positive_entry_frames']+=1
                matching=[e for e in row['carrier_site_entries'] if e['tick']>tick
                          and tuple(e['outside'])==tuple(outside) and tuple(e['inside'])==tuple(inside)]
                if not matching:
                    counts['positive_frames_without_matching_future_carrier_entry']+=1
                    cases.append(dict(round=row['round'],tick=int(tick),outside=outside.tolist(),inside=inside.tolist()))
        FIXED_COLLECTION_OUTPUT.write_text(json.dumps(dict(
            scope='actual_carrier_crossing_diagnostic_not_fixed_entry_counterfactual',
            seed=FIXED_COLLECTION_SEED,source_hashes=source['hashes'],optimizer_updates=0,
            counts=counts,examples=cases,rounds=rounds),ensure_ascii=False,indent=2),encoding='utf-8')
        print('Frozen collection:',counts)
        print('Saved:',FIXED_COLLECTION_OUTPUT)


if __name__=='__main__':
    main()
