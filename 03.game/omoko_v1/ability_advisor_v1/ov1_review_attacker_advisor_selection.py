"""Exploratory calibration of frozen weights; compare fresh matches without training."""

import copy
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import (
    ROOT, DEFAULT_MODEL_PATH, FixedAbilityAdvisor, sha256_file,
)
from omoko_v1.ability_advisor_v1.ov1_train_attacker_ability_advisor import (
    deduplicate_samples, split_matches, select_groups, operating_report,
)
from omoko_v1.ability_advisor_v1.ov1_collect_attacker_ability_advisor import play_match
from omoko_v1.ability_advisor_v1.ov1_evaluate_attacker_ability_usage import AbilityUsageMonitor
from omoko_v1.ability_advisor_v1.ov1_evaluate_attacker_ability_advisor import summarize, compare_summaries

DEFAULT_SOURCE = ROOT / 'data' / 'datasets' / '702d396fc72149d9a4b094d5f1869f80.parts'
DEFAULT_OUTPUT = ROOT / 'logs' / 'ov1_attacker_advisor_selection_trial.json'
DEFAULT_MATCHES = 2
DEFAULT_ROUNDS = 12
DEFAULT_OPPONENT = 'frc_v1'
DEFAULT_SEED = 700000
DEFAULT_SPLIT_SEED = 0
DEFAULT_RUN_MATCHES = True
MAX_OUTPUT_BYTES = 99_000_000


def run():
    torch.set_num_threads(1)
    digest = sha256_file(DEFAULT_MODEL_PATH)
    reference = FixedAbilityAdvisor()
    rows = []
    for path in sorted(DEFAULT_SOURCE.glob('*.part')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('event') == 'sample':
                    rows.append(row)
    if not rows:
        raise ValueError('diagnostic shards are unavailable; configure DEFAULT_SOURCE')
    folds = split_matches(deduplicate_samples(rows), DEFAULT_SPLIT_SEED)

    def scores(rows):
        inputs = torch.tensor(np.asarray([r['features'] for r in rows]), dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(reference.model(inputs)).tolist()

    proposals = select_groups(folds['validation'], scores(folds['validation']))
    groups = copy.deepcopy(reference.checkpoint['groups'])
    changed = {}
    # This is calibration of an already certified frozen model, not recertification
    # from incomplete shards. Unsupported groups retain their exact existing policy.
    for key, proposal in proposals.items():
        original = groups.get(key, {})
        if original.get('validated') and proposal['validated']:
            before = original.get('skip_threshold', reference.checkpoint['skip_threshold'])
            original['skip_threshold'] = proposal['skip_threshold']
            if before != proposal['skip_threshold']:
                changed[key] = dict(before=before, after=proposal['skip_threshold'])
    test_scores = scores(folds['test'])
    report = dict(
        model_sha256=digest, source=str(DEFAULT_SOURCE), changed_thresholds=changed,
        limitation='Calibration-only diagnostic using unfinished shards without provenance. '
                   'No neural-network training or adoption. Original training overlap cannot be excluded. '
                   'Fresh matches use separate seeds; they do not measure the future retrained model.',
        fold_samples={name:len(values) for name,values in folds.items()},
        offline_current=operating_report(folds['test'], test_scores, reference.checkpoint['groups']),
        offline_calibrated=operating_report(folds['test'], test_scores, groups),
        matches={}, opponent=DEFAULT_OPPONENT, seed=DEFAULT_SEED)
    results = {'current':[], 'calibration_only':[]}
    if DEFAULT_RUN_MATCHES:
        for trial in range(DEFAULT_MATCHES):
            for arm in results:
                advisor = FixedAbilityAdvisor()
                if arm == 'calibration_only':
                    advisor.checkpoint['groups'] = copy.deepcopy(groups)
                monitor = AbilityUsageMonitor(advisor)
                outcome = play_match(DEFAULT_OPPONENT, DEFAULT_SEED + trial, DEFAULT_ROUNDS,
                                     advisor, tick_observer=monitor.observe_tick)
                outcome['actual_usage'] = monitor.report()
                results[arm].append(outcome)
                print(f'match={trial + 1}/{DEFAULT_MATCHES} arm={arm} '
                      f'actual_usage={json.dumps(outcome["actual_usage"].get("all", {}))}', flush=True)
                if sha256_file(DEFAULT_MODEL_PATH) != digest:
                    raise RuntimeError('reference model changed during trial')
        report['matches'] = {arm:summarize(values) for arm,values in results.items()}
        report['comparison'] = compare_summaries(report['matches']['current'],
                                                report['matches']['calibration_only'])
    data = json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
    if len(data) > MAX_OUTPUT_BYTES:
        raise ValueError('trial report exceeds configured size limit')
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_bytes(data)
    print('offline_current', report['offline_current']['totals'])
    print('offline_calibrated', report['offline_calibrated']['totals'])
    print('report=' + str(DEFAULT_OUTPUT))


if __name__ == '__main__':
    run()
