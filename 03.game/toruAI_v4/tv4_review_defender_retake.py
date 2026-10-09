"""Read-only real-engine retake review; no optimizer or model promotion."""
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
REVIEW_SEEDS = (904400042, 904400142, 904400242)
REVIEW_PRESETS = ('Eine Kleine', 'SUPES', 'BBL')
REVIEW_MODES = ('legacy_teacher', 'combat_teacher', 'combat_learned')
MODEL_DIRECTORY = HERE / 'data' / 'best'
OUTPUT_DIRECTORY = HERE / 'logs' / 'retake_combat_review'
TORCH_THREADS = 1
REVIEW_SAVED_CASES = True
CASES_DIRECTORY = HERE / 'data' / 'retake_cases'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-only', action='store_true')
    args = parser.parse_args()
    from toruAI_v4.tv4_scenario import Scenario
    from toruAI_v4.tv4_defender_controller import load_analyses, load_policy
    from toruAI_v4.tv4_defender_policy import DefenderDQN, OBS_DIM
    from toruAI_v4.tv4_retake_combat import initialize_retake, RETAKE_OBS_DIM, retake_combat_schema
    from toruAI_v4.tv4_train_defender_search import rollout, summarize_defender
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    analyses, hashes = load_analyses(scenario, (OPPONENT,), MODEL_DIRECTORY)
    search, _ = load_policy(MODEL_DIRECTORY/OPPONENT/'search_best.pt', 'search', scenario)
    legacy, corrected, model_hashes = {}, {}, {}
    # Snapshot each source once, even if a running trainer replaces its file.
    for side in ('L', 'R'):
        payload = (MODEL_DIRECTORY/OPPONENT/f'retake_{side}_best.pt').read_bytes()
        saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        if saved['opponent'] != OPPONENT or saved['site'] != side or saved['schema']['board'] != scenario.grid.tolist():
            raise ValueError('Retake review checkpoint opponent/site/map mismatch')
        if saved['schema']['obs_dim'] != OBS_DIM:
            raise ValueError('This comparison requires the previous 505-feature retake checkpoint')
        legacy[side] = DefenderDQN(OBS_DIM)
        legacy[side].load_state_dict(saved['model'])
        corrected[side] = DefenderDQN(RETAKE_OBS_DIM)
        initialize_retake(corrected[side], legacy[side])
        legacy[side].eval(); corrected[side].eval()
        model_hashes[side] = hashlib.sha256(payload).hexdigest()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for mode in (() if args.cases_only else REVIEW_MODES):
        records = []
        enabled = mode != 'legacy_teacher'
        teacher = mode.endswith('teacher')
        for i, seed in enumerate(REVIEW_SEEDS):
            preset = REVIEW_PRESETS[i % len(REVIEW_PRESETS)]
            print(f'Review {mode} {OPPONENT} {preset} seed={seed}', flush=True)
            rounds, _ = rollout(OPPONENT, preset, scenario, search, corrected if enabled else legacy,
                analyses, 'retake', seed, training=teacher, teacher_probability=float(teacher), epsilon=0.,
                retake_combat_enabled=enabled)
            records.extend({'seed': seed, **r} for r in rounds)
        report = dict(mode=mode, opponent=OPPONENT, checkpoint_hashes=model_hashes,
            analysis_hashes=hashes, schema=retake_combat_schema(), metrics=summarize_defender(records), rounds=records,
            note='No training. Teacher modes validate demonstrations, not learned-policy performance.')
        (OUTPUT_DIRECTORY/f'{mode}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({'mode': mode, 'metrics': report['metrics']}, ensure_ascii=False), flush=True)
    if REVIEW_SAVED_CASES:
        from toruAI_v4.tv4_collect_retake import read_rows
        indexed = read_rows(CASES_DIRECTORY/'cases.jsonl')
        records = []
        for side in ('L', 'R'):
            row = next(r for r in indexed if r['opponent'] == OPPONENT and r['site'] == side)
            print(f'Review saved-case restore {OPPONENT} site={side}', flush=True)
            rounds, samples = rollout(OPPONENT, row['preset'], scenario, search, corrected, analyses,
                'retake', REVIEW_SEEDS[0], training=True, teacher_probability=1., epsilon=0.,
                initial_case=CASES_DIRECTORY/row['file'])
            if not samples or any(len(t)!=9 or len(t[0])!=RETAKE_OBS_DIM or len(t[3])!=RETAKE_OBS_DIM for t in samples):
                raise ValueError('Restored-case retake replay shape mismatch')
            records.extend(rounds)
        report = dict(mode='saved_case_teacher_smoke', metrics=summarize_defender(records), rounds=records,
                      note='Replay construction only; no optimizer update or model save.')
        (OUTPUT_DIRECTORY/'saved_case_teacher_smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({'mode': report['mode'], 'metrics': report['metrics']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
