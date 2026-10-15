"""Fit one shared adviser; split entire matches to keep past-round history out of other folds."""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

import numpy as np
import torch
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import (
    ROOT, VERSION, LABEL_VERSION, WALL_LOS_VERSION, DEFAULT_MODEL_PATH,
    AbilityJudge, feature_names,
)
from omoko_v1.ability_advisor_v1.ov1_collect_attacker_ability_advisor import route_metadata

DEFAULT_DATA_DIR = ROOT / 'data' / 'datasets'
DEFAULT_OUTPUT = DEFAULT_MODEL_PATH
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_SEED = 0
DEFAULT_SKIP_THRESHOLD = 0.10
DEFAULT_MIN_VALIDATION_ROUNDS = 20
DEFAULT_MAX_MISSED_RATE = 0.05
DEFAULT_PATIENCE = 10
DEFAULT_SKIP_INCOMPLETE_DATASETS = True


def load_samples(directory,skip_incomplete=DEFAULT_SKIP_INCOMPLETE_DATASETS):
    rows = []
    provenance = {}
    for path in sorted(Path(directory).glob('*.jsonl')):
        lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        try:
            completed = bool(lines) and json.loads(lines[-1]).get('event') == 'complete'
        except (ValueError,AttributeError):
            completed = False
        if not completed:
            if skip_incomplete:
                print(f'[SKIP] incomplete dataset: {path}',flush=True)
                continue
            raise ValueError(f'incomplete dataset: {path}')
        records = [json.loads(line) for line in lines]
        header = records[0]
        if (header.get('version') != VERSION or header.get('label_version') != LABEL_VERSION
                or header.get('wall_los_version') != WALL_LOS_VERSION
                or header.get('feature_names') != feature_names()):
            raise ValueError(f'dataset feature/label contract mismatch: {path}')
        current = route_metadata(tuple(header['scenarios']))
        if header['scenarios'] != current:
            raise ValueError(f'carry model or scenario changed; recollect {path}')
        provenance.update(current)
        for row in records[1:-1]:
            if row.get('event') != 'sample':
                continue
            features = np.asarray(row['features'],dtype=np.float32)
            if features.shape != (len(feature_names()),) or not np.isfinite(features).all() or row['label'] not in (0,1):
                raise ValueError(f'invalid sample in {path}')
            rows.append(row)
    if not rows:
        raise ValueError('no completed datasets; run the collector first')
    # Retried requests must not inflate a fixed point's round support.
    unique = {(r['match_id'],r['round_number'],r['group']):r for r in rows}
    return list(unique.values()),provenance


def split_matches(rows,seed):
    by_opponent = defaultdict(set)
    for row in rows:
        by_opponent[row['opponent']].add(row['match_id'])
    assignment = {}
    rng = random.Random(seed)
    for opponent, matches in sorted(by_opponent.items()):
        matches = sorted(matches)
        if len(matches) < 5:
            raise ValueError(f'{opponent}: at least 5 complete matches required')
        rng.shuffle(matches)
        held = max(1,int(len(matches)*0.2))
        for index,match in enumerate(matches):
            assignment[match] = 'test' if index < held else 'validation' if index < 2*held else 'train'
    return {name:[r for r in rows if assignment[r['match_id']] == name]
            for name in ('train','validation','test')}


def classify_report(rows,scores,threshold):
    groups = defaultdict(list)
    for row,score in zip(rows,scores):
        groups[row['group']].append((row,float(score)))
    result = {}
    for key,values in groups.items():
        positive = sum(row['label'] for row,_ in values)
        direct_index = feature_names().index('direct_relevance')
        skipped = [(row,score) for row,score in values if score < threshold
                   and not (row.get('features') and row['features'][direct_index] > 0)]
        missed = sum(row['label'] for row,_ in skipped)
        result[key] = dict(samples=len(values),rounds=len({(row['match_id'],row['round_number']) for row,_ in values}),
            positives=positive,skipped=len(skipped),missed_positives=missed,
            missed_rate=missed/positive if positive else 0.,
            unnecessary_skip_precision=1-missed/len(skipped) if skipped else None)
    return result


def safe_group(report,min_rounds,max_missed):
    return (report.get('rounds',0) >= min_rounds and report.get('skipped',0) >= 5
            and report['missed_rate'] <= max_missed
            and report['unnecessary_skip_precision'] >= 1-max_missed)


def tensors(rows):
    return (torch.tensor(np.asarray([r['features'] for r in rows]),dtype=torch.float32),
            torch.tensor([r['label'] for r in rows],dtype=torch.float32))


def train(data_dir=DEFAULT_DATA_DIR,output=DEFAULT_OUTPUT,epochs=DEFAULT_EPOCHS,seed=DEFAULT_SEED):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    rows,provenance = load_samples(data_dir)
    folds = split_matches(rows,seed)
    torch.manual_seed(seed)
    random.seed(seed)
    model = AbilityJudge()
    optimizer = torch.optim.Adam(model.parameters(),lr=DEFAULT_LEARNING_RATE)
    x,y = tensors(folds['train'])
    vx,vy = tensors(folds['validation'])
    best_loss,best_state,stale = float('inf'),None,0
    for epoch in range(1,epochs+1):
        model.train()
        indices = torch.randperm(len(x))
        for batch in indices.split(DEFAULT_BATCH_SIZE):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[batch]),y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(model(vx),vy))
        print(f'epoch={epoch}/{epochs} validation_loss={validation_loss:.5f}',flush=True)
        if validation_loss < best_loss:
            best_loss,best_state,stale = validation_loss,copy.deepcopy(model.state_dict()),0
        else:
            stale += 1
        if stale >= DEFAULT_PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    reports = {}
    for name in ('validation','test'):
        inputs,_ = tensors(folds[name])
        with torch.no_grad():
            scores = torch.sigmoid(model(inputs)).tolist()
        reports[name] = classify_report(folds[name],scores,DEFAULT_SKIP_THRESHOLD)
    groups = {}
    for key in set(reports['validation']) | set(reports['test']):
        validation,test = reports['validation'].get(key,{}),reports['test'].get(key,{})
        groups[key] = dict(validated=safe_group(validation,DEFAULT_MIN_VALIDATION_ROUNDS,DEFAULT_MAX_MISSED_RATE)
                          and safe_group(test,DEFAULT_MIN_VALIDATION_ROUNDS,DEFAULT_MAX_MISSED_RATE),
                          validation=validation,test=test)
    checkpoint = dict(version=VERSION,label_version=LABEL_VERSION,wall_los_version=WALL_LOS_VERSION,
        feature_names=feature_names(),scenarios=provenance,skip_threshold=DEFAULT_SKIP_THRESHOLD,
        groups=groups,model_state_dict=model.state_dict(),seed=seed,validation_loss=best_loss,
        split='whole_match_60_20_20',fold_samples={k:len(v) for k,v in folds.items()})
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    torch.save(checkpoint,temporary)
    temporary.replace(output)
    report_path = output.with_suffix('.report.json')
    report_path.write_text(json.dumps({k:v for k,v in checkpoint.items() if k != 'model_state_dict'},
                                      ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'model={output} validated_groups={sum(v["validated"] for v in groups.values())}/{len(groups)}',flush=True)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs',type=int,default=DEFAULT_EPOCHS)
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args.data_dir,args.output,args.epochs,args.seed)


if __name__ == '__main__':
    main()
