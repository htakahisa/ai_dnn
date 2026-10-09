"""Read-only review of trained retake v2 and deadline improvements."""
from pathlib import Path
import sys
import json
import io
import hashlib
import argparse
import torch

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
OPPONENT = 'frc_v1'
REVIEW_MODES = ('baseline', 'candidate', 'baseline_teacher', 'candidate_teacher')
MODEL_DIRECTORY = HERE/'data'/'best'
OUTPUT_DIRECTORY = HERE/'logs'/'retake_deadline_review'
TORCH_THREADS = 1
REVIEW_SAVED_CASES = True
CASES_DIRECTORY = HERE/'data'/'retake_cases'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', choices=REVIEW_MODES, default=REVIEW_MODES)
    parser.add_argument('--cases-only', action='store_true')
    args = parser.parse_args()
    from toruAI_v4.tv4_scenario import Scenario
    from toruAI_v4.tv4_defender_controller import load_policy, load_analyses
    from toruAI_v4.tv4_defender_policy import DefenderDQN
    from toruAI_v4.tv4_retake_combat import initialize_retake, RETAKE_OBS_DIM
    from toruAI_v4.tv4_train_defender_search import rollout, summarize_defender
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    analyses, _ = load_analyses(scenario, (OPPONENT,), MODEL_DIRECTORY)
    search, _ = load_policy(MODEL_DIRECTORY/OPPONENT/'search_best.pt', 'search', scenario)
    models, candidates, signatures, plans = {}, {}, {}, None
    for side in ('L', 'R'):
        payload = (MODEL_DIRECTORY/OPPONENT/f'retake_{side}_best.pt').read_bytes()
        saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        if saved['schema']['obs_dim'] != 519 or saved['schema']['board'] != scenario.grid.tolist():
            raise ValueError('Deadline comparison requires combat v2 checkpoints (519 features)')
        models[side] = DefenderDQN(519)
        models[side].load_state_dict(saved['model']); models[side].eval()
        candidates[side] = DefenderDQN(RETAKE_OBS_DIM)
        initialize_retake(candidates[side], models[side]); candidates[side].eval()
        signatures[side] = dict(hash=hashlib.sha256(payload).hexdigest(), adopted_set=saved['completed_sets'])
        plans = saved['evaluation_plan']
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if 'baseline' not in args.modes:
        reference = json.loads((OUTPUT_DIRECTORY/'baseline.json').read_text(encoding='utf8'))
        if reference['models'] != signatures:
            raise ValueError('Source best changed since baseline; run a new full comparison')
    for mode in (() if args.cases_only else args.modes):
        records = []
        for _, preset, seed in plans:
            print(f'Review {mode} {OPPONENT} {preset} seed={seed}', flush=True)
            enabled = mode.startswith('candidate')
            teacher = mode.endswith('teacher')
            rounds, _ = rollout(OPPONENT, preset, scenario, search, candidates if enabled else models, analyses,
                'retake', seed, retake_deadline_enabled=enabled, training=teacher, teacher_probability=float(teacher), epsilon=0.)
            records.extend(rounds)
        report = dict(mode=mode, opponent=OPPONENT, models=signatures, metrics=summarize_defender(records), rounds=records)
        report['note'] = 'No model update. candidate uses zero-extended v2 weights; candidate_teacher is demonstration validation.'
        (OUTPUT_DIRECTORY/f'{mode}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(dict(mode=mode,metrics=report['metrics']), ensure_ascii=False), flush=True)
    if REVIEW_SAVED_CASES:
        from toruAI_v4.tv4_collect_retake import read_rows
        indexed = read_rows(CASES_DIRECTORY/'cases.jsonl')
        records = []
        for side in ('L', 'R'):
            row = next(r for r in indexed if r['opponent']==OPPONENT and r['site']==side)
            rounds, transitions = rollout(OPPONENT, row['preset'], scenario, search, candidates, analyses, 'retake', plans[0][2],
                initial_case=CASES_DIRECTORY/row['file'], training=True, teacher_probability=1.)
            if not transitions or any(len(t)!=9 or len(t[0])!=RETAKE_OBS_DIM or len(t[3])!=RETAKE_OBS_DIM for t in transitions):
                raise ValueError('New retake case replay shape mismatch')
            records.extend(rounds)
        report = dict(mode='saved_case_teacher', metrics=summarize_defender(records), rounds=records)
        (OUTPUT_DIRECTORY/'saved_case_teacher.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(dict(mode=report['mode'],metrics=report['metrics']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
