"""Read-only detailed review of the trained v18 checkpoint's six failure cases.

Run from touyama_v3: python tv3_review_attacker_set5_failures.py
No optimizer, training, or model files are written.
"""
from pathlib import Path
import json
import tv3_review_attacker_timeout as trace

HERE=Path(__file__).resolve().parent
PLANT_CHECKPOINT=HERE/'data/attacker_plant/frc_v1/evaluation_checkpoints/set_000005.pt'
EXPECTED_PLANT_SET=5
EXPECTED_PLANT_VERSION=18
REVIEW_CASES=((804000142,1,8),(804000042,13,13),(804000042,13,15),
              (804000142,13,17),(804000242,1,3),(804000242,13,15))
OUTPUT_DIRECTORY=HERE/'logs/attacker_route_risk_review'
REUSE_MATCHING_TRACES=True


def main():
    trace.SOURCE_PLANT_CHECKPOINT=PLANT_CHECKPOINT
    trace.EXPECTED_PLANT_SET=EXPECTED_PLANT_SET
    source=trace.load_review_source(trace.Scenario())
    if (source['plant'].execution_version!=EXPECTED_PLANT_VERSION
            or source['plant_set']!=EXPECTED_PLANT_SET):
        raise ValueError('Unexpected plant set/version; update review constants explicitly')
    digest=source['hashes']['plant']
    for seed,start,round_number in REVIEW_CASES:
        trace.SEED,trace.START_ROUND,trace.TARGET_ROUND=seed,start,round_number
        trace.OUTPUT_PATH=OUTPUT_DIRECTORY/f'v18_set5_seed{seed}_r{round_number}_decisions.json'
        if REUSE_MATCHING_TRACES and trace.OUTPUT_PATH.exists():
            cached=json.loads(trace.OUTPUT_PATH.read_text(encoding='utf-8'))
            if (cached['source_hashes']['plant']==digest
                    and cached.get('source_version')==EXPECTED_PLANT_VERSION
                    and cached['seed']==seed and cached['start_round']==start
                    and cached['round']==round_number and not cached.get('choice_overrides')
                    and not cached.get('plan_order_overrides')
                    and not cached.get('candidate_version')
                    and not cached.get('candidate_checkpoint')
                    and all(not cached.get(k) for k in ('net_progress_clock','forced_teacher_target',
                        'forced_diagnostic_site','reassign_blocked_plant_helper','release_deadline_route_reserve',
                        'restore_unique_progress_edge','restore_all_progress_edges','defer_initial_assembly',
                        'confirmed_empty_view','restore_confirmed_flash_progress'))):
                print('Reused:',trace.OUTPUT_PATH,flush=True)
                continue
        trace.main()


if __name__=='__main__':
    main()
