"""Compare an extra unnecessary-use gate using saved reports, without training."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import ROOT, sha256_file

DEFAULT_REPORT = ROOT / 'data' / 'ov1_attacker_ability_advisor.report.json'
DEFAULT_OUTPUT = ROOT / 'logs' / 'ov1_attacker_advisor_validation_review.json'
DEFAULT_MAX_UNNECESSARY_USE_RATES = (0.05, 0.10, 0.20, 0.30, 0.50)
MAX_OUTPUT_BYTES = 1_000_000


def counts(report):
    """Recover the confusion matrix, including the direct-observation override."""
    negatives = report.get('samples', 0) - report.get('positives', 0)
    saved_negatives = report.get('skipped', 0) - report.get('missed_positives', 0)
    allowed_negatives = negatives - saved_negatives
    if not 0 <= saved_negatives <= negatives:
        raise ValueError('invalid saved report counts')
    return dict(negatives=negatives, saved_negatives=saved_negatives,
                unnecessary_allowed=allowed_negatives,
                unnecessary_use_rate=allowed_negatives / negatives if negatives else None)


def passes_extra_gate(group, limit):
    # No negatives means no evidence for this new criterion.
    return bool(group['validated']) and all(
        (rate := counts(group.get(fold, {}))['unnecessary_use_rate']) is not None
        and rate <= limit for fold in ('validation', 'test'))


def operational_totals(groups, enabled):
    """Unvalidated groups allow every candidate, as in the real adviser."""
    total = dict(samples=0, negatives=0, positives=0, unnecessary_allowed=0,
                 necessary_saved=0, saved_negatives=0)
    for key, group in groups.items():
        report = group.get('test', {})
        matrix = counts(report)
        total['samples'] += report.get('samples', 0)
        total['negatives'] += matrix['negatives']
        total['positives'] += report.get('positives', 0)
        if enabled[key]:
            total['unnecessary_allowed'] += matrix['unnecessary_allowed']
            total['necessary_saved'] += report.get('missed_positives', 0)
            total['saved_negatives'] += matrix['saved_negatives']
        else:
            total['unnecessary_allowed'] += matrix['negatives']
    total['unnecessary_use_rate'] = total['unnecessary_allowed'] / total['negatives']
    return total


def review():
    report = json.loads(DEFAULT_REPORT.read_text(encoding='utf-8'))
    groups = report['groups']
    baseline = {k: bool(g['validated']) for k, g in groups.items()}
    result = dict(source=str(DEFAULT_REPORT), source_sha256=sha256_file(DEFAULT_REPORT),
                  definition='unnecessary_allowed / teacher_negative_samples',
                  scope='Retrospective fixed-candidate test counts, not a match replay or win-rate estimate. '
                        'Weights, scores, threshold and existing gates are unchanged. '
                        'Extra gate must pass in both validation and test, as existing certification does.',
                  baseline=dict(validated_groups=sum(baseline.values()),
                                test=operational_totals(groups, baseline)),
                  comparisons={}, targets={})
    for limit in DEFAULT_MAX_UNNECESSARY_USE_RATES:
        enabled = {k: passes_extra_gate(g, limit) for k, g in groups.items()}
        result['comparisons'][str(limit)] = dict(
            validated_groups=sum(enabled.values()),
            rejected_groups=[k for k in groups if baseline[k] and not enabled[k]],
            test=operational_totals(groups, enabled))
    for key, group in sorted(groups.items()):
        if '|frc_v1|FLASH|' in key and key.endswith(('13,31', '11,16')):
            result['targets'][key] = dict(
                validated=group['validated'],
                folds={fold:dict(**group.get(fold, {}), **counts(group.get(fold, {})))
                       for fold in ('validation', 'test')},
                passes={str(limit):passes_extra_gate(group, limit)
                        for limit in DEFAULT_MAX_UNNECESSARY_USE_RATES})
    data = json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8')
    if len(data) > MAX_OUTPUT_BYTES:
        raise ValueError('review output exceeds configured limit')
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_bytes(data)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    review()
