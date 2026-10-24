"""Read-only evaluation of existing analysis candidates with one pinned plant."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
BEST_DIRECTORY=HERE/'data'/'best'
LATEST_PATH=HERE/'data'/'attacker_analysis'/OPPONENT/'latest.pt'
SEEDS=(904000042,904000142,904000242)
OUTPUT_PATH=HERE/'logs'/'attacker_route_risk_review'/'analysis_candidates.json'
REPLAY_OUTPUT_PATH=HERE/'logs'/'attacker_route_risk_review'/'analysis_replay_comparison.json'
TORCH_THREADS=1
EXPLORATION_SCOPE='all_routes'
EXPLORATION_PROBE_STEPS=24
EXPLORATION_PROBE_TICKS=40

import hashlib
import json
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_learn_attacker_analysis import load_analysis
from touyama_v3.tv3_collect_attacker_analysis import play_block
from touyama_v3.tv3_train_attacker_analysis import summarize,prediction_metrics,best_rank


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    latest,latest_encoder,state=load_analysis(LATEST_PATH,scenario,OPPONENT)
    if state['config'].get('collection_source_hashes')!=source['hashes']:
        raise ValueError('Latest was collected with a different pinned executor pair')
    saved=load_checkpoint(source['paths']['analysis'])
    config={**saved['config'], 'collection_executor':'fixed_plant',
        'exploration_scope':EXPLORATION_SCOPE,'exploration_probe_steps':EXPLORATION_PROBE_STEPS,
        'exploration_probe_ticks':EXPLORATION_PROBE_TICKS,'exploration_round_start_only':True}
    from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder
    candidates=(('adopted',source['analysis'],AttackerEncoder(scenario,source['analysis'].execution_version),source['hashes']['analysis']),
                ('latest',latest,latest_encoder,hashlib.sha256(LATEST_PATH.read_bytes()).hexdigest()))
    output=dict(scope='diagnostic_existing_models_not_publication_or_training',
        scenario=scenario.signature,executor_hashes=source['hashes'],seeds=SEEDS,
        config=config,candidates={},optimizer_updates=0)
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    replay=[{k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in row.items()} for row in state['replay']]
    same_inputs=all(row['observations'].shape[1]==len(encoder.fields) and
                    row['routes'].shape[1]==len(encoder.route_fields)
                    for _,_,encoder,_ in candidates for row in replay)
    replay_result=dict(scope='same_inputs_from_latest_training_replay_not_holdout',rounds=len(replay),
                       frames=sum(len(r['outcomes']) for r in replay),executor_hashes=source['hashes'])
    if same_inputs:
        for name,model,encoder,digest in candidates:
            replay_result[name]=prediction_metrics(model,replay,encoder.fields)
    else:
        replay_result['skipped']='Candidate input contracts differ; do not coerce inputs'
    REPLAY_OUTPUT_PATH.write_text(json.dumps(replay_result,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,model,encoder,digest in candidates:
        selected=[]; exploratory=[]; samples=[]
        for seed in SEEDS:
            rounds,_=play_block(OPPONENT,scenario,model,config,seed,0.,preset_name='Touyama Gaming',plant_policy=source['plant'])
            selected.extend(dict(seed=seed,**r) for r in rounds)
            rounds,batch=play_block(OPPONENT,scenario,model,config,seed+500000,1.,
                preset_name='Touyama Gaming',plant_policy=source['plant'])
            exploratory.extend(rounds); samples.extend(batch)
            evaluation=dict(selected=summarize(selected),random=summarize(exploratory),
                            prediction=prediction_metrics(model,samples,encoder.fields))
            output['candidates'][name]=dict(hash=digest,completed_seeds=len(selected)//12,
                trained_rounds=model.trained_rounds,evaluation=evaluation,legacy_rank=best_rank(evaluation),rounds=selected)
            OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
            print(name,seed,'selected',sum(r['planted'] for r in selected),'/',len(selected),
                  'brier',evaluation['prediction']['brier'],flush=True)
    print('Saved:',OUTPUT_PATH,flush=True)


if __name__=='__main__':
    main()
