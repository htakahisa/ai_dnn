"""Compare full attacking matches with and without fixed-ability advice."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

import torch
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import (
    ROOT, MAPS, OPPONENTS, ABILITIES, DEFAULT_MODEL_PATH, FixedAbilityAdvisor, sha256_file,
    DeployedAbilityAdvisor, DEFAULT_FLASH_SMOKE_MODEL_PATH,
)
from omoko_v1.ability_advisor_v1.ov1_collect_attacker_ability_advisor import play_match
from omoko_v1.ability_advisor_v1.ov1_opponent_processes import run_opponent_tasks
from omoko_v1.ability_advisor_v1.ov1_evaluate_attacker_ability_usage import AbilityUsageMonitor

DEFAULT_MATCHES_PER_OPPONENT = 10
DEFAULT_ROUNDS_PER_MATCH = 12
DEFAULT_OPPONENTS = OPPONENTS
DEFAULT_MAPS = MAPS
DEFAULT_MODEL = DEFAULT_MODEL_PATH
DEFAULT_SEED = 100000  # Independent of collection seeds.
DEFAULT_OUTPUT = ROOT / 'logs' / 'ov1_attacker_ability_advisor_evaluation.json'
DEFAULT_WORKERS = 6  # Independent opponent comparisons; 1 runs serially.
DEFAULT_CANDIDATE_MODEL = ROOT / 'data' / 'ov1_attacker_ability_advisor_candidate.pt'
DEFAULT_DEPLOYED_FLASH_SMOKE_MODEL = DEFAULT_FLASH_SMOKE_MODEL_PATH
MAX_OUTPUT_BYTES = 99_000_000


def summarize(results):
    rounds = sum(r['rounds'] for r in results)
    plants = [p for r in results for p in r['charges_at_plant']]
    usage = {}
    for result in results:
        for key, stats in result.get('actual_usage', {}).items():
            combined = usage.setdefault(key, dict.fromkeys(stats, 0))
            for field, value in stats.items():
                combined[field] += value
    return dict(matches=len(results),rounds=rounds, actual_usage=usage,
        win_rate=sum(r['attacker_wins'] for r in results)/rounds if rounds else 0.,
        plant_rate=len(plants)/rounds if rounds else 0.,
        avg_charges_at_plant={a:sum(p[a] for p in plants)/len(plants) if plants else None for a in ABILITIES},
        advised_uses=sum(r['decisions']['use'] for r in results),
        advised_saves=sum(r['decisions']['save'] for r in results))


def compare_summaries(reference, candidate):
    usage = {}
    for key in sorted(set(reference['actual_usage']) | set(candidate['actual_usage'])):
        before = reference['actual_usage'].get(key, {})
        after = candidate['actual_usage'].get(key, {})
        usage[key] = {}
        for field in ('casts', 'teacher_unnecessary_casts', 'zero_enemy_effect_projectiles'):
            previous, current = before.get(field, 0), after.get(field, 0)
            usage[key][field] = dict(reference=previous, candidate=current,
                                     reduction=previous-current,
                                     reduction_rate=(previous-current)/previous if previous else None)
    return dict(win_rate_difference=candidate['win_rate']-reference['win_rate'],
                plant_rate_difference=candidate['plant_rate']-reference['plant_rate'], actual_usage=usage)


def evaluate_opponent(task):
    (opponent,index,model_path,digest,matches,rounds,maps,seed,candidate_path,candidate_digest,
     deployed_path,deployed_digest) = task
    arms = {'baseline':[],'advised':[]}
    if candidate_path is not None:
        arms['candidate'] = []
    if deployed_path is not None:
        arms['deployed'] = []
    for trial in range(matches):
        for name in arms:
            path = candidate_path if name == 'candidate' else model_path
            advisor = (DeployedAbilityAdvisor(model_path, deployed_path) if name == 'deployed' else
                       FixedAbilityAdvisor(path if name != 'baseline' else None,
                                           enabled=name != 'baseline'))
            if name != 'baseline' and advisor.model is None:
                raise ValueError('adviser model could not be loaded')
            monitor = AbilityUsageMonitor(advisor)
            result = play_match(opponent,seed+index*matches+trial,rounds,advisor,maps,
                                tick_observer=monitor.observe_tick)
            result['actual_usage'] = monitor.report()
            arms[name].append(result)
        if sha256_file(model_path) != digest:
            raise RuntimeError('adviser weights changed during evaluation')
        if candidate_path is not None and sha256_file(candidate_path) != candidate_digest:
            raise RuntimeError('candidate weights changed during evaluation')
        if deployed_path is not None and sha256_file(deployed_path) != deployed_digest:
            raise RuntimeError('adopted FLASH/SMOKE weights changed during evaluation')
        print(f'evaluate {opponent} match={trial+1}/{matches}',flush=True)
    summary = {name:summarize(results) for name,results in arms.items()}
    if candidate_path is not None:
        summary['candidate_vs_advised'] = compare_summaries(summary['advised'], summary['candidate'])
    if deployed_path is not None:
        summary['deployed_vs_advised'] = compare_summaries(summary['advised'], summary['deployed'])
        if candidate_path is not None:
            summary['candidate_vs_deployed'] = compare_summaries(summary['deployed'], summary['candidate'])
    print(opponent,json.dumps(summary,ensure_ascii=False),flush=True)
    return opponent,summary


def evaluate(model_path=DEFAULT_MODEL,matches=DEFAULT_MATCHES_PER_OPPONENT,
             rounds=DEFAULT_ROUNDS_PER_MATCH,opponents=DEFAULT_OPPONENTS,
             maps=DEFAULT_MAPS,seed=DEFAULT_SEED,output=DEFAULT_OUTPUT,workers=DEFAULT_WORKERS,
             candidate_path=None, deployed_path=None):
    if workers < 1:
        raise ValueError('workers must be positive')
    if matches < 1 or not 1 <= rounds <= 12:
        raise ValueError('positive match count and 1..12 attacking rounds required')
    if not opponents or any(n not in OPPONENTS for n in opponents):
        raise ValueError('select known opponents')
    if not maps or any(n not in MAPS for n in maps):
        raise ValueError('select Omoko role/site contexts')
    model_path = Path(model_path).resolve()
    digest = sha256_file(model_path)
    candidate_path = Path(candidate_path).resolve() if candidate_path is not None else None
    candidate_digest = sha256_file(candidate_path) if candidate_path is not None else None
    deployed_path = Path(deployed_path).resolve() if deployed_path is not None else None
    deployed_digest = sha256_file(deployed_path) if deployed_path is not None else None
    report = dict(model_sha256=digest,candidate_sha256=candidate_digest,
                  deployed_flash_smoke_sha256=deployed_digest,seed=seed,opponents={})
    tasks = [(opponent,index,model_path,digest,matches,rounds,tuple(maps),seed,candidate_path,candidate_digest,
              deployed_path,deployed_digest)
             for index,opponent in enumerate(opponents)]
    report['opponents'] = dict(run_opponent_tasks(evaluate_opponent,tasks,workers))
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    data = json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8')
    if len(data) > MAX_OUTPUT_BYTES:
        raise ValueError('evaluation report exceeds configured size limit')
    output.write_bytes(data)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--matches',type=int,default=DEFAULT_MATCHES_PER_OPPONENT)
    parser.add_argument('--rounds',type=int,default=DEFAULT_ROUNDS_PER_MATCH)
    parser.add_argument('--opponents',nargs='+',choices=OPPONENTS,default=DEFAULT_OPPONENTS)
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--workers',type=int,default=DEFAULT_WORKERS)
    parser.add_argument('--candidate',type=Path,default=DEFAULT_CANDIDATE_MODEL)
    parser.add_argument('--deployed-model',type=Path,default=DEFAULT_DEPLOYED_FLASH_SMOKE_MODEL)
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluate(args.model,args.matches,args.rounds,args.opponents,DEFAULT_MAPS,args.seed,args.output,args.workers,
             candidate_path=args.candidate, deployed_path=args.deployed_model)


if __name__ == '__main__':
    main()
