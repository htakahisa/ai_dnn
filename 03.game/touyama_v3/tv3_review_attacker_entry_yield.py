"""Evaluate goal repair from exact saved weights; no training or promotion."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
OPPONENT='frc_v1'
SOURCE_DIRECTORY=HERE/'data'/'best'
EXPECTED_PLANT_SET=10
EXPECTED_PLANT_VERSION=16
CANDIDATE_VERSION=18
TORCH_THREADS=1
OUTPUT_DIRECTORY=HERE/'logs'/'attacker_route_risk_review'
OUTPUT_PATH=OUTPUT_DIRECTORY/'set10_deadline_goal_72round_evaluation.json'
ROUND_PATH=OUTPUT_DIRECTORY/'set10_deadline_goal_72round_rounds.jsonl'

import hashlib,json,torch
from touyama_v3.tv3_guard_runtime import load_sources,read_checkpoint
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_plant import PlantDQN,policy_schema
from touyama_v3.tv3_train_attacker_plant import evaluate


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),SOURCE_DIRECTORY,SOURCE_DIRECTORY)[OPPONENT]
    saved,digest=read_checkpoint(SOURCE_DIRECTORY/OPPONENT/'attacker_plant_best.pt')
    if (source['plant_set']!=EXPECTED_PLANT_SET or source['plant'].execution_version!=EXPECTED_PLANT_VERSION
            or digest!=source['hashes']['plant']):
        raise ValueError('Saved model changed; comparison must be relabelled')
    candidate=PlantDQN(CANDIDATE_VERSION)
    candidate.load_state_dict(source['plant'].state_dict());candidate.eval()
    OUTPUT_DIRECTORY.mkdir(parents=True,exist_ok=True)
    # A fresh report avoids mixing rounds from repeated candidate revisions.
    ROUND_PATH.write_text('',encoding='utf8')
    result=evaluate(OPPONENT,scenario,source['analysis'],candidate,saved['config'],saved['evaluation']['seeds'],
                    record_path=ROUND_PATH,completed=0)
    output=dict(source_hashes=source['hashes'],source_set=source['plant_set'],
        candidate_version=CANDIDATE_VERSION,optimizer_updates=0,candidate_trained=False,
        schema_hash=hashlib.sha256(json.dumps(policy_schema(scenario,CANDIDATE_VERSION),sort_keys=True).encode()).hexdigest(),
        baseline=saved['evaluation'],candidate=result)
    OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    for label,evaluation in [('baseline',saved['evaluation']),('candidate',result)]:
        metrics=evaluation['selected']
        print(label,{k:metrics[k] for k in ('plants','advantaged_plants','competitive_plants','timeouts')},
              'sites',{s:{k:r[k] for k in ('rounds','plants','timeouts')} for s,r in metrics['by_site'].items()},flush=True)
    print('Saved:',OUTPUT_PATH,flush=True)


if __name__=='__main__':
    main()
