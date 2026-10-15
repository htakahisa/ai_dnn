"""Compare full attacking matches with and without fixed-ability advice."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

import torch
from concon_v1.ability_advisor_v1.co1_learn_attacker_ability_advisor import (
    ROOT, MAPS, OPPONENTS, ABILITIES, DEFAULT_MODEL_PATH, FixedAbilityAdvisor, sha256_file,
)
from concon_v1.ability_advisor_v1.co1_collect_attacker_ability_advisor import play_match
from concon_v1.ability_advisor_v1.co1_opponent_processes import run_opponent_tasks

DEFAULT_MATCHES_PER_OPPONENT = 10
DEFAULT_ROUNDS_PER_MATCH = 12
DEFAULT_OPPONENTS = OPPONENTS
DEFAULT_MAPS = MAPS
DEFAULT_MODEL = DEFAULT_MODEL_PATH
DEFAULT_SEED = 100000  # Independent of collection seeds.
DEFAULT_OUTPUT = ROOT / 'logs' / 'co1_attacker_ability_advisor_evaluation.json'
DEFAULT_WORKERS = 6  # Independent opponent comparisons; 1 runs serially.


def summarize(results):
    rounds = sum(r['rounds'] for r in results)
    plants = [p for r in results for p in r['charges_at_plant']]
    return dict(matches=len(results),rounds=rounds,
        win_rate=sum(r['attacker_wins'] for r in results)/rounds if rounds else 0.,
        plant_rate=len(plants)/rounds if rounds else 0.,
        avg_charges_at_plant={a:sum(p[a] for p in plants)/len(plants) if plants else None for a in ABILITIES},
        advised_uses=sum(r['decisions']['use'] for r in results),
        advised_saves=sum(r['decisions']['save'] for r in results))


def evaluate_opponent(task):
    opponent,index,model_path,digest,matches,rounds,maps,seed = task
    arms = {'baseline':[],'advised':[]}
    for trial in range(matches):
        for name in arms:
            advisor = FixedAbilityAdvisor(model_path if name == 'advised' else None)
            if name == 'advised' and advisor.model is None:
                raise ValueError('adviser model could not be loaded')
            result = play_match(opponent,seed+index*matches+trial,rounds,advisor,maps)
            arms[name].append(result)
        if sha256_file(model_path) != digest:
            raise RuntimeError('adviser weights changed during evaluation')
        print(f'evaluate {opponent} match={trial+1}/{matches}',flush=True)
    summary = {name:summarize(results) for name,results in arms.items()}
    print(opponent,json.dumps(summary,ensure_ascii=False),flush=True)
    return opponent,summary


def evaluate(model_path=DEFAULT_MODEL,matches=DEFAULT_MATCHES_PER_OPPONENT,
             rounds=DEFAULT_ROUNDS_PER_MATCH,opponents=DEFAULT_OPPONENTS,
             maps=DEFAULT_MAPS,seed=DEFAULT_SEED,output=DEFAULT_OUTPUT,workers=DEFAULT_WORKERS):
    if workers < 1:
        raise ValueError('workers must be positive')
    if matches < 1 or not 1 <= rounds <= 12:
        raise ValueError('positive match count and 1..12 attacking rounds required')
    if not opponents or any(n not in OPPONENTS for n in opponents):
        raise ValueError('select known opponents')
    if not maps or any(n not in MAPS for n in maps):
        raise ValueError('select A1-A4 routes')
    model_path = Path(model_path).resolve()
    digest = sha256_file(model_path)
    report = dict(model_sha256=digest,seed=seed,opponents={})
    tasks = [(opponent,index,model_path,digest,matches,rounds,tuple(maps),seed)
             for index,opponent in enumerate(opponents)]
    report['opponents'] = dict(run_opponent_tasks(evaluate_opponent,tasks,workers))
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--matches',type=int,default=DEFAULT_MATCHES_PER_OPPONENT)
    parser.add_argument('--rounds',type=int,default=DEFAULT_ROUNDS_PER_MATCH)
    parser.add_argument('--opponents',nargs='+',choices=OPPONENTS,default=DEFAULT_OPPONENTS)
    parser.add_argument('--maps',nargs='+',choices=MAPS,default=DEFAULT_MAPS)
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--workers',type=int,default=DEFAULT_WORKERS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluate(args.model,args.matches,args.rounds,args.opponents,args.maps,args.seed,args.output,args.workers)


if __name__ == '__main__':
    main()
