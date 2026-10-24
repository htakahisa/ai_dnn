"""Paired 72-round evaluation of left goal repair; no learning or model writes."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# 通常設定：この冒頭を確認し、touyama_v3からオプションなしで実行する。
OPPONENT = 'frc_v1'
SOURCE_CHECKPOINT = HERE/'data'/'best'/'custom_profiles'/'28f115895deba593'/'frc_v1'/'attacker_plant_best.pt'
EXPECTED_SET = 5
EXPECTED_SOURCE_SHA = '864cf058d336267876f87bf0ae4cae51afb2871e7d3407ac69a118d1c79ee9b1'
EVALUATION_SEEDS = (804000042, 804000142, 804000242)
EVALUATION_START_ROUNDS = (1, 13)
PRESET = 'Touyama Gaming'
TORCH_THREADS = 1
OUTPUT_PATH = HERE/'logs'/'custom_extensions'/'left_goal_yield_72round_comparison.json'

import hashlib
import json
from unittest.mock import patch
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController as Controller
from touyama_v3.tv3_review_attacker_timeout import load_review_source, json_value
from touyama_v3.tv3_train_attacker_plant import rollout, summarize_plant
from touyama_v3.tv3_learn_attacker_plant_left_goal_yield import LeftGoalYield
from touyama_v3.tv3_learn_attacker_fire_support import public_threats


class GoalRepairReview:
    """Allocate independent round-local state for each rollout controller."""
    def __init__(self, original):
        self.original = original
        self.states = {}
        self.events = []

    def goal_for(self, controller, snapshot, ally):
        goal = self.original(controller, snapshot, ally)
        state = self.states.setdefault(controller, dict(repair=LeftGoalYield(), key=None, goals={}))
        if controller.route.site != 'L':
            state['repair'].reset()
            state['key'] = None
            return goal
        if snapshot.key != state['key']:
            goals = {a.slot: controller._assigned_goal_for(snapshot, a)
                     for a in snapshot.allies if a.alive}
            repair = state['repair']
            state['goals'] = repair.update(controller.scenario, snapshot, goals, site='L',
                threats=public_threats(snapshot, controller.analysis_encoder.history.tracks),
                coordinated_wait=bool(controller.site_entry_state and controller.site_entry_state['wait']))
            if repair.events and repair.events[-1]['tick'] == snapshot.tick:
                self.events.append(dict(round=snapshot.round_number, **repair.events[-1]))
            state['key'] = snapshot.key
        return state['goals'].get(ally.slot, goal)


def main():
    outputs = {name: OUTPUT_PATH.with_name(f'{OUTPUT_PATH.stem}_{name}.jsonl')
               for name in ('baseline','candidate')}
    if any(p.exists() for p in (OUTPUT_PATH, *outputs.values())):
        raise ValueError('Choose a new OUTPUT_PATH; comparison outputs already exist')
    torch.set_num_threads(TORCH_THREADS)
    digest = hashlib.sha256(SOURCE_CHECKPOINT.read_bytes()).hexdigest()
    if digest != EXPECTED_SOURCE_SHA:
        raise ValueError('Source checkpoint changed; check EXPECTED_SOURCE_SHA')
    scenario = Scenario()
    source = load_review_source(scenario, SOURCE_CHECKPOINT)
    if source['plant_set'] != EXPECTED_SET:
        raise ValueError('Unexpected source set')
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location='cpu', weights_only=True)
    common_path = HERE/'data'/'best'/OPPONENT/'attacker_plant_best.pt'
    common_digest = hashlib.sha256(common_path.read_bytes()).hexdigest()
    config = checkpoint['config']
    if list(EVALUATION_SEEDS) != checkpoint['evaluation']['seeds']:
        raise ValueError('Seeds differ from the saved evaluation')
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rounds, events = {name: [] for name in outputs}, []
    original = Controller.goal_for
    for seed in EVALUATION_SEEDS:
        for start in EVALUATION_START_ROUNDS:
            for name in outputs:
                review = GoalRepairReview(original)
                if name == 'candidate':
                    def repaired(controller, snapshot, ally):
                        return review.goal_for(controller, snapshot, ally)
                    with patch.object(Controller, 'goal_for', repaired):
                        records, _ = rollout(OPPONENT, scenario, source['analysis'], source['plant'],
                            config, seed, preset_name=PRESET, start_round=start)
                else:
                    records, _ = rollout(OPPONENT, scenario, source['analysis'], source['plant'],
                        config, seed, preset_name=PRESET, start_round=start)
                labelled = [dict(seed=seed,start_round=start,**r) for r in records]
                rounds[name].extend(labelled)
                with outputs[name].open('a', encoding='utf-8') as out:
                    for r in labelled:
                        out.write(json.dumps(r,ensure_ascii=False,default=json_value)+'\n')
                events.extend(dict(seed=seed,start_round=start,**e) for e in review.events)
                print(json.dumps(dict(arm=name,seed=seed,start_round=start,rounds=len(records),
                    plants=sum(r['planted'] for r in records),timeouts=sum(r['reason']=='timeout' for r in records),
                    repairs=len(review.events)),ensure_ascii=False),flush=True)
    metrics = {name: summarize_plant(rows) for name, rows in rounds.items()}
    lookup = {(r['seed'],r['round']): r for r in rounds['baseline']}
    fields = ('target_site','planted','tick','alive','enemy_alive','remaining_hp','reason')
    changes = []
    for after in rounds['candidate']:
        before = lookup[(after['seed'],after['round'])]
        if any(before[k] != after[k] for k in fields):
            changes.append(dict(seed=after['seed'],round=after['round'],
                before={k:before[k] for k in fields},after={k:after[k] for k in fields}))
    unchanged = (hashlib.sha256(SOURCE_CHECKPOINT.read_bytes()).hexdigest() == digest
                 and hashlib.sha256(common_path.read_bytes()).hexdigest() == common_digest)
    if not unchanged:
        raise ValueError('Model files changed during read-only evaluation')
    summary = dict(source=str(SOURCE_CHECKPOINT),source_hashes=source['hashes'],
        candidate_source_sha=hashlib.sha256((HERE/'tv3_learn_attacker_plant_left_goal_yield.py').read_bytes()).hexdigest(),
        completed_sets=EXPECTED_SET,optimizer_updates=0,models_unchanged=unchanged,
        production_activation=False,seeds=list(EVALUATION_SEEDS),start_rounds=list(EVALUATION_START_ROUNDS),
        preset=PRESET,teacher_probability=0,epsilon=0,route_exploration=0,
        metrics=metrics,repair_events=events,changed_rounds=changes)
    OUTPUT_PATH.write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=json_value),encoding='utf-8')
    print('Saved:',OUTPUT_PATH,flush=True)


if __name__ == '__main__':
    main()
