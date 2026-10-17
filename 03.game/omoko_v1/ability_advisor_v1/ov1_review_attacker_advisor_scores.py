"""Read-only score diagnosis; unfinished shards are exploratory, not held-out tests."""

import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import (
    ROOT, DEFAULT_MODEL_PATH, feature_names, sha256_file,
)
from omoko_v1.ability_advisor_v1.ov1_advisor_attacker_context import preflight_default_advisor

DEFAULT_DATA_DIR = ROOT / 'data' / 'datasets'
DEFAULT_OUTPUT = ROOT / 'logs' / 'ov1_attacker_advisor_score_review.json'
DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.50)
DEFAULT_OPPONENT = 'frc_v1'
DEFAULT_POINTS = ('13,31', '11,16')
MAX_REPORT_BYTES = 1_000_000


def is_target(key):
    return f'|{DEFAULT_OPPONENT}|FLASH|' in key and key.split('|')[-1] in DEFAULT_POINTS


def distribution(values):
    return dict(zip(('min', 'p25', 'median', 'p75', 'p90', 'max'),
                    np.quantile(values, (0, .25, .5, .75, .9, 1)).tolist())) if len(values) else {}


def retry_report(rows, allowed):
    """Compare the last request with all requests; permissions are not casts."""
    groups = defaultdict(lambda: defaultdict(list))
    for row, use in zip(rows, allowed):
        groups[row['group']][row['match_id'], row['round_number']].append((row, bool(use)))
    result = {}
    for key, rounds in sorted(groups.items()):
        negatives = [items for items in rounds.values()
                     if not any(row['label'] for row, _ in items)]
        result[key] = dict(
            raw_candidates=sum(map(len, rounds.values())), rounds=len(rounds),
            all_negative_rounds=len(negatives),
            all_negative_rounds_last_allowed=sum(items[-1][1] for items in negatives),
            all_negative_rounds_any_allowed=sum(any(use for _, use in items) for items in negatives),
            rounds_with_changing_labels=sum(len({row['label'] for row, _ in items}) > 1
                                           for items in rounds.values()))
    return result


def review():
    torch.set_num_threads(1)
    judge = preflight_default_advisor()
    checkpoint = judge.checkpoint
    result = dict(model_sha256=sha256_file(DEFAULT_MODEL_PATH),
                  route_compatibility='passed_all_four',
                  limitation='Only unfinished shards remain; no provenance header or completion marker. '
                             'These are exploratory inputs, not the original training/validation/test data.',
                  saved_fold_samples=checkpoint['fold_samples'],
                  saved_validation_loss=checkpoint['validation_loss'],
                  saved_target_reports={k:v for k,v in checkpoint['groups'].items() if is_target(k)},
                  datasets={})
    direct_index = feature_names().index('direct_relevance')
    for directory in sorted(DEFAULT_DATA_DIR.glob('*.parts')):
        unique = {}
        target_requests = []
        raw_count = invalid = 0
        for path in sorted(directory.glob('*.part')):
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        invalid += 1
                        continue
                    if row.get('event') == 'sample':
                        raw_count += 1
                        if is_target(row['group']):
                            target_requests.append(row)
                        unique[row['match_id'], row['round_number'], row['group']] = row
        rows = list(unique.values())
        if not rows:
            continue
        x = torch.tensor(np.asarray([r['features'] for r in rows]), dtype=torch.float32)
        with torch.no_grad():
            scores = torch.sigmoid(judge.model(x)).numpy()
        labels = np.asarray([r['label'] for r in rows], dtype=bool)
        direct = x[:, direct_index].numpy() > 0
        validated = np.asarray([checkpoint['groups'].get(r['group'], {}).get('validated', False)
                                for r in rows])
        dataset = dict(raw_samples=raw_count, unique_samples=len(rows), invalid_lines=invalid,
                       thresholds={}, targets={})
        if target_requests:
            requests = torch.tensor(np.asarray([r['features'] for r in target_requests]),
                                    dtype=torch.float32)
            with torch.no_grad():
                request_scores = torch.sigmoid(judge.model(requests)).numpy()
            request_allowed = [
                row['features'][direct_index] > 0
                or not checkpoint['groups'].get(row['group'], {}).get('validated', False)
                or score >= checkpoint['skip_threshold']
                for row, score in zip(target_requests, request_scores)]
            dataset['retry_sequences'] = retry_report(target_requests, request_allowed)
        for threshold in DEFAULT_THRESHOLDS:
            use = direct | ~validated | (scores >= threshold)
            dataset['thresholds'][str(threshold)] = dict(
                unnecessary_allowed=int((use & ~labels).sum()),
                necessary_saved=int((~use & labels).sum()),
                necessary_total=int(labels.sum()),
                note='Uses existing validated gates; a changed threshold requires renewed validation.')
        for key in sorted({r['group'] for r in rows if is_target(r['group'])}):
            mask = np.asarray([r['group'] == key for r in rows])
            dataset['targets'][key] = dict(
                samples=int(mask.sum()), positives=int(labels[mask].sum()),
                score_distribution=distribution(scores[mask]),
                direct_allowed=int((mask & direct).sum()),
                thresholds={str(t):dict(
                    unnecessary_allowed=int((mask & ~labels & (direct | ~validated | (scores >= t))).sum()),
                    necessary_saved=int((mask & labels & ~(direct | ~validated | (scores >= t))).sum()))
                    for t in DEFAULT_THRESHOLDS})
        result['datasets'][directory.name] = dataset
        print(directory.name, json.dumps(dataset, ensure_ascii=False), flush=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8')
    if len(encoded) > MAX_REPORT_BYTES:
        raise ValueError('diagnostic report exceeds configured limit')
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_bytes(encoded)
    print('report=' + str(DEFAULT_OUTPUT))


if __name__ == '__main__':
    review()
