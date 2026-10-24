"""Non-learning left-goal repair review; production models/settings are untouched."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# 通常設定：touyama_v3から python tv3_review_attacker_plant_left_goal_yield.py
SOURCE_CHECKPOINT = HERE/'data'/'best'/'custom_profiles'/'28f115895deba593'/'frc_v1'/'attacker_plant_best.pt'
EXPECTED_SET = 5
SEED = 804000042
START_ROUND = 1
TARGET_ROUND = 9
ENABLE_LEFT_GOAL_REPAIR = True
OUTPUT_PATH = HERE/'logs'/'custom_extensions'/'left_goal_yield_seed042_round9_confirmed.json'

import json
from unittest.mock import patch
from touyama_v3 import tv3_review_attacker_timeout as trace
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController as Controller
from touyama_v3.tv3_learn_attacker_plant_left_goal_yield import LeftGoalYield
from touyama_v3.tv3_learn_attacker_fire_support import public_threats


def main():
    if OUTPUT_PATH.exists():
        raise ValueError(f'Choose a new OUTPUT_PATH: {OUTPUT_PATH}')
    repair = LeftGoalYield()
    original = Controller.goal_for
    cached_key, replacements = None, {}
    def goal_for(controller, snapshot, ally):
        nonlocal cached_key, replacements
        goal = original(controller, snapshot, ally)
        if not ENABLE_LEFT_GOAL_REPAIR or snapshot.round_number != TARGET_ROUND or controller.route.site != 'L':
            return goal
        key = snapshot.key
        if key != cached_key:
            goals = {a.slot: controller._assigned_goal_for(snapshot, a)
                     for a in snapshot.allies if a.alive}
            replacements = repair.update(controller.scenario, snapshot, goals, site='L',
                threats=public_threats(snapshot, controller.analysis_encoder.history.tracks),
                coordinated_wait=bool(controller.site_entry_state and controller.site_entry_state['wait']))
            cached_key = key
        return replacements.get(ally.slot, goal)
    with patch.multiple(trace, SOURCE_PLANT_CHECKPOINT=SOURCE_CHECKPOINT, EXPECTED_PLANT_SET=EXPECTED_SET,
                        SEED=SEED, START_ROUND=START_ROUND, TARGET_ROUND=TARGET_ROUND, OUTPUT_PATH=OUTPUT_PATH), \
            patch.object(Controller, 'goal_for', goal_for):
        trace.main()
    result = json.loads(OUTPUT_PATH.read_text(encoding='utf-8'))
    result.update(optimizer_updates=0, goal_repair_enabled=ENABLE_LEFT_GOAL_REPAIR,
                  left_goal_repair_events=repair.events, production_activation=False)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=trace.json_value), encoding='utf-8')
    print({k: result['terminal'][k] for k in ('planted','tick','alive','enemy_alive','remaining_hp')})
    print('Goal repairs:', repair.events)


if __name__ == '__main__':
    main()
