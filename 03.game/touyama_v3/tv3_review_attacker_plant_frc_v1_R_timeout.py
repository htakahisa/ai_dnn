"""Matched-prefix deadline teacher probe; no learning or checkpoint writes."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SOURCE_CHECKPOINT = HERE/'data'/'best'/'custom_profiles'/'3e2fa49fec166d75'/'frc_v1'/'attacker_plant_best.pt'
EXPECTED_SET = 5
SEED = 804000142
START_ROUND = 1
TARGET_ROUND = 11
APPLY_DEADLINE_TEACHER = True  # 診断用。対象ラウンドの保持者の期限教師だけを実行。
OUTPUT_PATH = HERE/'logs'/'custom_extensions'/'revision3_review_seed142_round11_probe.json'
EXTENSION_MODULE = 'tv3_custom_train_attacker_plant_frc_v1_R_v3'

import json
from unittest.mock import patch
from touyama_v3 import tv3_review_attacker_timeout as trace
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
from touyama_v3.tv3_model_extensions import build_extensions, ExtensionContext


def main():
    if OUTPUT_PATH.exists():
        raise ValueError(f'Review output already exists: {OUTPUT_PATH}; choose a new OUTPUT_PATH')
    pipeline = build_extensions((EXTENSION_MODULE,), role='attacker_plant', opponent='frc_v1')
    module = pipeline.entries[0][0]
    original = TouyamaV3AttackerPlantController.prepare_team_tick
    overrides = {}
    def prepare(controller):
        target = controller.game.current_round == TARGET_ROUND
        if target:
            controller.custom_pipeline = pipeline
        original(controller)
        if not target or not APPLY_DEADLINE_TEACHER or controller.snapshot is None:
            return
        for _, (chosen, inputs, ally) in controller.plans.items():
            context = ExtensionContext('attacker_plant', 'frc_v1', controller.route.site,
                                       controller.snapshot, ally, inputs.goal)
            if module._deadline(context) is not None and inputs.teacher != chosen:
                overrides[(controller.snapshot.tick, ally.slot)] = int(inputs.teacher)
    with patch.multiple(trace, SOURCE_PLANT_CHECKPOINT=SOURCE_CHECKPOINT, EXPECTED_PLANT_SET=EXPECTED_SET,
                        SEED=SEED, START_ROUND=START_ROUND, TARGET_ROUND=TARGET_ROUND,
                        OUTPUT_PATH=OUTPUT_PATH, DIAGNOSTIC_CHOICE_OVERRIDES=overrides), \
            patch.object(TouyamaV3AttackerPlantController, 'prepare_team_tick', prepare):
        trace.main()
    result = json.loads(OUTPUT_PATH.read_text(encoding='utf-8'))
    result.update(optimizer_updates=0, learned_candidate=False, deadline_extension=pipeline.contract,
                  deadline_teacher_probe=APPLY_DEADLINE_TEACHER)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['terminal'], ensure_ascii=False)[:1200])


if __name__ == '__main__':
    main()
