"""Validate route-entry binding with existing analysis/plant and cached matched baseline."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
BEST_DIRECTORY=HERE/'data'/'best'
LATEST_PATH=HERE/'data'/'attacker_analysis'/OPPONENT/'latest.pt'
BASELINE_PATH=HERE/'logs'/'attacker_route_risk_review'/'analysis_candidates.json'
OUTPUT_DIRECTORY=HERE/'logs'/'attacker_route_risk_review'
OUTPUT_PREFIX='entry_binding_latest'
TORCH_THREADS=1
BIND_CHANGED_ENTRY_ONLY=True  # Isolate stale-goal correction; preserve initial preparation criteria.

import json
import hashlib
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_learn_attacker_analysis import load_analysis
from touyama_v3.tv3_train_attacker_plant import rollout
from touyama_v3.tv3_train_attacker_analysis import summarize


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    output_path=OUTPUT_DIRECTORY/f'{OUTPUT_PREFIX}_{"changes_only" if BIND_CHANGED_ENTRY_ONLY else "full"}.json'
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    latest,_,state=load_analysis(LATEST_PATH,scenario,OPPONENT)
    baseline=json.loads(BASELINE_PATH.read_text(encoding='utf-8'))
    digest=hashlib.sha256(LATEST_PATH.read_bytes()).hexdigest()
    if (baseline['executor_hashes']!=source['hashes'] or baseline['scenario']!=scenario.signature
        or baseline['candidates']['latest']['hash']!=digest
        or state['config'].get('collection_source_hashes')!=source['hashes']):
        raise ValueError('Baseline or pinned executor does not match current candidate')
    seeds=baseline['seeds']
    config=load_checkpoint(source['paths']['plant'])['config']
    original,_=rollout(OPPONENT,scenario,latest,source['plant'],config,seeds[0],
                       preset_name='Touyama Gaming',training=False)
    reference=[r for r in baseline['candidates']['latest']['rounds'] if r['seed']==seeds[0]]
    fields=('round','planted','tick','alive','remaining_hp','reason','site')
    if len(original)!=len(reference) or any(any(a[k]!=b[k] for k in fields) for a,b in zip(original,reference)):
        raise RuntimeError('No-op rollout differs from cached baseline; do not compare outcomes')
    result=dict(scope='entry_binding_with_existing_v3_not_v4_training_or_model_publication',
        source_hashes=source['hashes'],analysis_hash=digest,optimizer_updates=0,
        bind_changed_entry_only=BIND_CHANGED_ENTRY_ONLY,
        baseline_parity_checked_seed=seeds[0],baseline_summary=baseline['candidates']['latest']['evaluation']['selected'],rounds=[])
    def starting(controller,round_number):
        controller.site_assembly.follow_route_entry=True
        controller.site_assembly.bind_changed_entry_only=BIND_CHANGED_ENTRY_ONLY
    for seed in seeds:
        rows,_=rollout(OPPONENT,scenario,latest,source['plant'],config,seed,
            preset_name='Touyama Gaming',training=False,round_start_callback=starting)
        result['rounds'].extend(dict(seed=seed,**r) for r in rows)
        result['summary']=summarize(result['rounds'])
        output_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(seed,'plants',sum(r['planted'] for r in rows),'/12',flush=True)
    print('Saved:',output_path,flush=True)


if __name__=='__main__':
    main()
