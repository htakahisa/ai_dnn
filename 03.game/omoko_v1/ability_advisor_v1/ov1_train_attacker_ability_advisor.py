"""Fit one shared adviser; split entire matches to keep past-round history out of other folds."""

import argparse
from collections import defaultdict
import copy
import json
import io
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
from omoko_v1.ability_advisor_v1.ov1_collect_attacker_ability_advisor import (
    route_metadata, delete_incompatible_datasets,
)
from omoko_v1.ability_advisor_v1.ov1_advisor_attacker_context import compatible_route_metadata

DEFAULT_DATA_DIR = ROOT / 'data' / 'datasets'
DEFAULT_OUTPUT = ROOT / 'data' / 'ov1_attacker_ability_advisor_candidate.pt'
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_SEED = 0
DEFAULT_SKIP_THRESHOLD = 0.10
DEFAULT_MIN_VALIDATION_ROUNDS = 20
DEFAULT_MAX_MISSED_RATE = 0.05
DEFAULT_PATIENCE = 10
DEFAULT_SKIP_INCOMPLETE_DATASETS = True
DEFAULT_THRESHOLD_CANDIDATES = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30,
                                0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
DEFAULT_MAX_MISSED_ROUND_RATE = 0.05
TRAINING_CONTRACT = 'all_decisions_validation_selection_v2'
MAX_OUTPUT_BYTES = 99_000_000


def load_samples(directory,skip_incomplete=DEFAULT_SKIP_INCOMPLETE_DATASETS):
    delete_incompatible_datasets(directory)
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
        if any(not compatible_route_metadata(metadata, current[name])
               for name, metadata in header['scenarios'].items()):
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
    return deduplicate_samples(rows),provenance


def deduplicate_samples(rows):
    # Preserve distinct decisions. Round support is counted separately below.
    unique = {}
    for row in rows:
        key = (row['match_id'], row['round_number'], row['group'], row.get('caster'),
               row.get('tick'), tuple(row['features']), row['label'])
        unique.setdefault(key, row)
    return list(unique.values())


def round_weights(rows):
    counts = defaultdict(int)
    keys = [(r['match_id'], r['round_number'], r['group']) for r in rows]
    for key in keys:
        counts[key] += 1
    weights = np.asarray([1 / counts[key] for key in keys], dtype=np.float32)
    return torch.from_numpy(weights / weights.mean())


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
        rounds = defaultdict(list)
        for row, score in values:
            direct = bool(row.get('features') and row['features'][direct_index] > 0)
            rounds[row['match_id'], row['round_number']].append((row['label'], direct or score >= threshold))
        negative_rounds = [v for v in rounds.values() if not any(label for label, _ in v)]
        positive_rounds = [v for v in rounds.values() if any(label for label, _ in v)]
        missed_rounds = sum(any(label and not use for label, use in v) for v in positive_rounds)
        negatives = len(values) - positive
        unnecessary = negatives - (len(skipped) - missed)
        result[key] = dict(samples=len(values),rounds=len({(row['match_id'],row['round_number']) for row,_ in values}),
            positives=positive,skipped=len(skipped),missed_positives=missed,
            missed_rate=missed/positive if positive else 0.,
            unnecessary_skip_precision=1-missed/len(skipped) if skipped else None,
            negatives=negatives, unnecessary_allowed=unnecessary,
            unnecessary_use_rate=unnecessary/negatives if negatives else None,
            negative_rounds=len(negative_rounds),
            unnecessary_use_rounds=sum(any(use for _, use in v) for v in negative_rounds),
            positive_rounds=len(positive_rounds), missed_positive_rounds=missed_rounds,
            skipped_rounds=sum(any(not use for _, use in v) for v in rounds.values()),
            missed_round_rate=missed_rounds/len(positive_rounds) if positive_rounds else 0.)
    return result


def safe_group(report,min_rounds,max_missed):
    return (report.get('rounds',0) >= min_rounds and report.get('skipped_rounds',report.get('skipped',0)) >= 5
            and report['missed_rate'] <= max_missed
            and report['unnecessary_skip_precision'] >= 1-max_missed)


def select_groups(rows, scores):
    """Select per-group thresholds on validation only; never select on test labels."""
    by_group = defaultdict(list)
    for row, score in zip(rows, scores):
        by_group[row['group']].append((row, score))
    selected = {}
    for key, values in by_group.items():
        inputs, probabilities = zip(*values)
        candidates = []
        for threshold in DEFAULT_THRESHOLD_CANDIDATES:
            report = classify_report(inputs, probabilities, threshold)[key]
            if (safe_group(report, DEFAULT_MIN_VALIDATION_ROUNDS, DEFAULT_MAX_MISSED_RATE)
                    and report['missed_round_rate'] <= DEFAULT_MAX_MISSED_ROUND_RATE):
                # Minimize whole-round opportunities for waste first, then individual permissions.
                candidates.append(((report['unnecessary_use_rounds'], report['unnecessary_allowed'],
                                    report['missed_positives'], -threshold), threshold, report))
        if candidates:
            _, threshold, report = min(candidates)
            selected[key] = dict(validated=True, skip_threshold=threshold, validation=report)
        else:
            selected[key] = dict(validated=False, skip_threshold=DEFAULT_SKIP_THRESHOLD,
                                 validation=classify_report(inputs, probabilities, DEFAULT_SKIP_THRESHOLD)[key])
    return selected


def operating_report(rows, scores, groups):
    """Evaluate actual gates, including use-all fallback, on every decision."""
    operational = []
    for row, score in zip(rows, scores):
        group = groups.get(row['group'], {})
        operational.append(float(not group.get('validated', False) or
                                 score >= group.get('skip_threshold', DEFAULT_SKIP_THRESHOLD)))
    reports = classify_report(rows, operational, .5)
    fields = ('samples', 'negatives', 'positives', 'unnecessary_allowed', 'missed_positives',
              'negative_rounds', 'unnecessary_use_rounds', 'positive_rounds', 'missed_positive_rounds')
    totals = {field: sum(report[field] for report in reports.values()) for field in fields}
    for name, numerator, denominator in (
        ('unnecessary_use_rate', 'unnecessary_allowed', 'negatives'),
        ('unnecessary_use_round_rate', 'unnecessary_use_rounds', 'negative_rounds'),
        ('missed_rate', 'missed_positives', 'positives')):
        totals[name] = totals[numerator] / totals[denominator] if totals[denominator] else 0.
    return dict(totals=totals, groups=reports)


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
    best_key, best_groups, best_epoch = None, None, None
    history = []
    weights = round_weights(folds['train'])
    validation_weights = round_weights(folds['validation'])
    for epoch in range(1,epochs+1):
        model.train()
        indices = torch.randperm(len(x))
        for batch in indices.split(DEFAULT_BATCH_SIZE):
            losses = torch.nn.functional.binary_cross_entropy_with_logits(model(x[batch]),y[batch], reduction='none')
            loss = (losses * weights[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(vx)
            validation_loss = float((torch.nn.functional.binary_cross_entropy_with_logits(
                logits,vy,reduction='none') * validation_weights).mean())
            scores = torch.sigmoid(logits).tolist()
        groups = select_groups(folds['validation'], scores)
        operating = operating_report(folds['validation'], scores, groups)['totals']
        selection_key = (operating['unnecessary_use_round_rate'], operating['unnecessary_use_rate'], validation_loss)
        history.append(dict(epoch=epoch, validation_loss=validation_loss,
                            validated_groups=sum(g['validated'] for g in groups.values()), operating=operating))
        print(f'epoch={epoch}/{epochs} validation_loss={validation_loss:.5f} '
              f'unnecessary_use_round_rate={selection_key[0]:.5f} '
              f'unnecessary_use_rate={selection_key[1]:.5f}',flush=True)
        if best_key is None or selection_key < best_key:
            best_loss,best_state,stale = validation_loss,copy.deepcopy(model.state_dict()),0
            best_key,best_groups,best_epoch = selection_key,copy.deepcopy(groups),epoch
        else:
            stale += 1
        if stale >= DEFAULT_PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    reports, operations = {}, {}
    groups = best_groups
    for name in ('validation','test'):
        inputs,_ = tensors(folds[name])
        with torch.no_grad():
            scores = torch.sigmoid(model(inputs)).tolist()
        operations[name] = operating_report(folds[name], scores, groups)
        by_group = defaultdict(list)
        for row, score in zip(folds[name], scores):
            by_group[row['group']].append((row, score))
        reports[name] = {}
        for key, values in by_group.items():
            inputs, probabilities = zip(*values)
            threshold = groups.get(key, {}).get('skip_threshold', DEFAULT_SKIP_THRESHOLD)
            reports[name][key] = classify_report(inputs, probabilities, threshold)[key]
    for key in set(reports['validation']) | set(reports['test']):
        group = groups.setdefault(key, dict(validated=False, skip_threshold=DEFAULT_SKIP_THRESHOLD))
        group.update(validation=reports['validation'].get(key, {}), test=reports['test'].get(key, {}))
    checkpoint = dict(version=VERSION,label_version=LABEL_VERSION,wall_los_version=WALL_LOS_VERSION,
        feature_names=feature_names(),scenarios=provenance,skip_threshold=DEFAULT_SKIP_THRESHOLD,
        groups=groups,model_state_dict=model.state_dict(),seed=seed,validation_loss=best_loss,
        split='whole_match_60_20_20',fold_samples={k:len(v) for k,v in folds.items()},
        training_contract=TRAINING_CONTRACT, best_epoch=best_epoch, epoch_history=history,
        selection='validation_only_minimize_unnecessary_use_rounds_then_candidates',
        sample_weighting='equal_total_weight_per_match_round_group',
        threshold_candidates=DEFAULT_THRESHOLD_CANDIDATES, operating=operations)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    buffer = io.BytesIO()
    torch.save(checkpoint,buffer)
    if buffer.tell() > MAX_OUTPUT_BYTES:
        raise ValueError('adviser checkpoint exceeds configured size limit')
    report_bytes = json.dumps({k:v for k,v in checkpoint.items() if k != 'model_state_dict'},
                             ensure_ascii=False,indent=2).encode('utf-8')
    if len(report_bytes) > MAX_OUTPUT_BYTES:
        raise ValueError('adviser report exceeds configured size limit')
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(output)
    report_path = output.with_suffix('.report.json')
    report_path.write_bytes(report_bytes)
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
