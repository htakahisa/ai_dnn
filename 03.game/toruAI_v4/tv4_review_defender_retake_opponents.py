"""Per-opponent paired retake diagnostics. No training or model writes."""
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
REVIEW_OPPONENTS = ('frc_v1', 'fnatic_v3', 'gc_v1', 'omoko_v1', 'toru_ai_v3', 'touyama_v2')
REVIEW_MODES = ('baseline', 'candidate')
MODEL_DIRECTORY = HERE/'data'/'best'
OUTPUT_DIRECTORY = HERE/'logs'/'retake_opponent_review'
TORCH_THREADS = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opponents', nargs='+', default=REVIEW_OPPONENTS, choices=REVIEW_OPPONENTS)
    args = parser.parse_args()
    from toruAI_v4.tv4_scenario import Scenario
    from toruAI_v4.tv4_defender_controller import load_policy, load_analyses
    from toruAI_v4.tv4_defender_policy import DefenderDQN
    from toruAI_v4.tv4_retake_combat import RETAKE_OBS_DIM
    from toruAI_v4.tv4_train_defender_search import rollout, summarize_defender
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    # Read all sources once, before any evaluations, to isolate active training.
    sources = {}
    for opponent in args.opponents:
        analyses, hashes = load_analyses(scenario, (opponent,), MODEL_DIRECTORY)
        search_bytes = (MODEL_DIRECTORY/opponent/'search_best.pt').read_bytes()
        search, _ = load_policy(io.BytesIO(search_bytes), 'search', scenario)
        models, metadata, plan = {}, {}, None
        for side in ('L', 'R'):
            payload = (MODEL_DIRECTORY/opponent/f'retake_{side}_best.pt').read_bytes()
            saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
            if saved['schema']['obs_dim'] != RETAKE_OBS_DIM or saved['schema']['board'] != scenario.grid.tolist():
                raise ValueError('Review requires compatible 528-feature retake models')
            if saved['opponent'] != opponent or saved['site'] != side or saved['analysis_hashes'][opponent] != hashes[opponent]:
                raise ValueError('Retake model opponent/site/analysis mismatch')
            if saved['frozen_search_hash'] != hashlib.sha256(search_bytes).hexdigest():
                raise ValueError('Retake/search hash mismatch')
            if plan is not None and plan != saved['evaluation_plan']:
                raise ValueError('Left/right evaluation plans differ')
            plan = saved['evaluation_plan']
            models[side] = DefenderDQN(RETAKE_OBS_DIM)
            models[side].load_state_dict(saved['model']); models[side].eval()
            metadata[side] = dict(hash=hashlib.sha256(payload).hexdigest(), adopted_set=saved['completed_sets'],
                                  previous_evaluation=saved['evaluation'])
        sources[opponent] = search, analyses, models, metadata, plan
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for opponent, (search, analyses, models, metadata, plan) in sources.items():
        for mode in REVIEW_MODES:
            records = []
            for _, preset, seed in plan:
                print(f'Review {opponent} {mode} {preset} seed={seed}', flush=True)
                rounds, _ = rollout(opponent, preset, scenario, search, models, analyses, 'retake', seed,
                                    legacy_dead_defuser_mask=mode=='baseline')
                records.extend(rounds)
            metrics = {side: summarize_defender([r for r in records if r['site']==side]) for side in ('L', 'R')}
            output = dict(opponent=opponent, mode=mode, models=metadata, plan=plan, metrics=metrics, rounds=records,
                note='Same weights. Baseline reproduces dead-defuser mask bug; no training or checkpoint writes.')
            (OUTPUT_DIRECTORY/f'{opponent}_{mode}.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf8')
            print(json.dumps(dict(opponent=opponent, mode=mode, metrics=metrics), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
