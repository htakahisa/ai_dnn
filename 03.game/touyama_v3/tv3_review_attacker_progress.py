"""Matched inference pilot; copies weights in memory, never trains or saves models."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
OPPONENT = 'frc_v1'
SOURCE_DIRECTORY = HERE / 'data' / 'best'
EXPECTED_PLANT_SET = 10
EXPECTED_PLANT_VERSION = 16
SEED = 804000142
TORCH_THREADS = 1
CANDIDATE_VERSION = 17  # Diagnostic weights copied from the saved model; no checkpoint promotion.
TEACHER_ONLY = False  # Diagnostic action examples only; no optimizer updates.
CARRIER_ONLY = False  # Applicable only to the rejected v14 movement experiment.
BASELINE_CACHE = HERE / 'logs' / 'attacker_route_risk_review' / 'set10_helper_goal_12round_pilot.json'
OUTPUT_PATH = BASELINE_CACHE

import json
import hashlib
import torch
from unittest.mock import patch
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController as Controller
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_learn_attacker_plant import PlantDQN,policy_schema
from touyama_v3.tv3_train_attacker_plant import rollout


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    source = load_sources(scenario, (OPPONENT,), SOURCE_DIRECTORY, SOURCE_DIRECTORY)[OPPONENT]
    if source['plant_set'] != EXPECTED_PLANT_SET or source['plant'].execution_version != EXPECTED_PLANT_VERSION:
        raise ValueError('Saved plant set/version differ from the declared comparison')
    candidate = PlantDQN(CANDIDATE_VERSION)
    candidate.load_state_dict(source['plant'].state_dict())
    candidate.eval()
    output = dict(seed=SEED, source_hashes=source['hashes'], plant_set=source['plant_set'],
                  optimizer_updates=0, candidate_trained=False, candidate_version=CANDIDATE_VERSION,
                  carrier_only=CARRIER_ONLY,teacher_only=TEACHER_ONLY, results={})
    output['candidate_schema_hash']=hashlib.sha256(json.dumps(
        policy_schema(scenario,CANDIDATE_VERSION),sort_keys=True).encode()).hexdigest()
    def started(controller,round_number):
        if TEACHER_ONLY:
            controller.training=True
            controller.teacher_probability=1.
            controller.epsilon=0.
    original = Controller._restrict_movement
    def restrict(controller, ally, inputs, reserved):
        version = controller.encoder.execution_version
        if CANDIDATE_VERSION==14 and CARRIER_ONLY and not ally.has_spike:
            controller.encoder.execution_version = 13
        try:
            original(controller, ally, inputs, reserved)
        finally:
            controller.encoder.execution_version = version
    cached = json.loads(BASELINE_CACHE.read_text(encoding='utf8')) if BASELINE_CACHE.exists() else None
    baseline_label=f'baseline_v{source["plant"].execution_version}'
    for label, model in [(baseline_label, source['plant']), (f'candidate_v{CANDIDATE_VERSION}', candidate)]:
        if (label == baseline_label and cached and cached['seed'] == SEED
                and cached['source_hashes'] == source['hashes'] and cached.get('teacher_only',False)==TEACHER_ONLY):
            rows = cached['results'][label]
        else:
            with patch.object(Controller, '_restrict_movement', restrict):
                rows, _ = rollout(OPPONENT, scenario, source['analysis'], model,
                              dict(max_round_steps=400, combat_diagnostics=False), SEED,
                              preset_name='Touyama Gaming',round_start_callback=started)
        output['results'][label] = rows
        print(label, [(r['round'], r['reason'], r['alive'], r['enemy_alive']) for r in rows], flush=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf8')
    print('Saved:', OUTPUT_PATH, flush=True)


if __name__ == '__main__':
    main()
