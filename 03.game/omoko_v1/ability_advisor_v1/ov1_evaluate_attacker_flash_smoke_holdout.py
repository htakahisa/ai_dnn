"""Predeclared fresh-seed comparison: candidate FLASH/SMOKE and reference RECON."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from omoko_v1.ability_advisor_v1.ov1_review_attacker_flash_outcomes import (
    run_seed, summarize, DEFAULT_MODEL_PATH, DEFAULT_CANDIDATE,
)
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import ROOT, sha256_file
from omoko_v1.ability_advisor_v1.ov1_opponent_processes import run_opponent_tasks
from omoko_v1.ability_advisor_v1.ov1_advisor_attacker_context import policy_digest, DEFAULT_POLICY_PATHS

DEFAULT_FIRST_SEED = 810000
DEFAULT_MATCHES = 20
DEFAULT_ARMS = ('current', 'flash_smoke')
DEFAULT_WORKERS = 4
DEFAULT_OUTPUT = ROOT / 'logs' / 'ov1_attacker_flash_smoke_holdout.json'
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_BOOTSTRAP_SAMPLES = 10000
MAX_OUTPUT_BYTES = 99_000_000


def fingerprints():
    return dict(reference=sha256_file(DEFAULT_MODEL_PATH), candidate=sha256_file(DEFAULT_CANDIDATE),
                attacker_policies=policy_digest(DEFAULT_POLICY_PATHS))


def main():
    seeds = tuple(range(DEFAULT_FIRST_SEED, DEFAULT_FIRST_SEED + DEFAULT_MATCHES))
    hashes = fingerprints()
    runs = run_opponent_tasks(run_seed, [(seed, DEFAULT_ARMS) for seed in seeds], DEFAULT_WORKERS)
    if hashes != fingerprints():
        raise RuntimeError('models changed during the fresh-seed comparison')
    summary = summarize(runs, DEFAULT_ARMS)
    paired = [dict(seed=run['seed'],
                   reference=run['arms']['current']['outcome']['attacker_wins'],
                   flash_smoke=run['arms']['flash_smoke']['outcome']['attacker_wins'],
                   rounds=run['arms']['current']['outcome']['rounds']) for run in runs]
    differences = np.asarray([r['flash_smoke']-r['reference'] for r in paired])
    rounds = np.asarray([r['rounds'] for r in paired])
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    indices = rng.integers(0, len(paired), size=(DEFAULT_BOOTSTRAP_SAMPLES, len(paired)))
    bootstrapped = differences[indices].sum(axis=1) / rounds[indices].sum(axis=1)
    comparison = dict(additional_wins=int(differences.sum()),
                      win_rate_difference=float(differences.sum()/rounds.sum()),
                      better_matches=int((differences > 0).sum()),
                      worse_matches=int((differences < 0).sum()),
                      tied_matches=int((differences == 0).sum()),
                      paired_match_bootstrap_95_interval=np.quantile(bootstrapped, [.025, .975]).tolist())
    report = dict(model_hashes=hashes, seeds=seeds, arms=DEFAULT_ARMS, summary=summary,
                  comparison=comparison, paired_matches=paired,
                  limitation='Twenty predeclared fresh-seed matches per arm. Bootstrap resamples paired '
                             'whole matches, preserving within-match round dependence. '
                             'Results apply to FRC; no other opponent was evaluated or model adopted.',
                  runs=runs)
    data = json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
    if len(data) > MAX_OUTPUT_BYTES:
        raise ValueError('holdout report exceeds configured size limit')
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_bytes(data)
    print(json.dumps(dict(summary=summary, comparison=comparison), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
