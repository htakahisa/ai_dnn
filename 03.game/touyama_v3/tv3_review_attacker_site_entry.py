"""Evaluate frozen plant best and inspect public entry/traffic; never train."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OPPONENT = "frc_v1"
BEST_DIRECTORY = HERE / "data" / "best"
EVALUATION_SEEDS = (804000142,)  # Existing best evaluation seed containing a timeout with five survivors.
PRESET = "Touyama Gaming"
OUTPUT_FILE = HERE / "logs" / "attacker_plant_review" / "frc_v1_site_entry_review.json"
TORCH_THREADS = 1
TAIL_TICKS = 12

import json
from collections import Counter, defaultdict, deque
import torch
from frc_v1.actions import MOVE_STEPS
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_plant import load_plant, load_frozen_analysis
from touyama_v3.tv3_train_attacker_plant import rollout, summarize_plant


class EntryReview:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.tail = defaultdict(lambda: deque(maxlen=TAIL_TICKS))

    def observe(self, controller):
        snapshot = controller.snapshot
        if not controller.plans:
            return  # Retain the last actual decisions when no route fits the clock.
        actors = []
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        for name, (selected, inputs, ally) in controller.plans.items():
            action = inputs.actions[selected]
            combat = inputs.combat
            actors.append(dict(slot=ally.slot, position=ally.position, hp=ally.hp,
                carrier=ally.has_spike, action=action.kind, goal=inputs.goal,
                teacher=inputs.actions[inputs.teacher].kind, combat=combat.reason,
                contacts=combat.contacts, supporters=combat.supporters,
                static_distance=int(inputs.distances[ally.position]),
                traffic_distance=int(inputs.traffic[ally.position]) if inputs.traffic is not None else None,
                site_distance=int(controller.scenario.site_dist[controller.route.site][ally.position]),
                legal_moves=sorted({inputs.actions[i].kind for i in range(40) if inputs.mask[i]})))
            counter = self.counts[snapshot.round_number]
            if action.kind in MOVE_STEPS and combat.contacts:
                counter['moving_under_contact'] += 1
            if ally.has_spike and action.kind == 'STAY' and combat.reason == 'advance':
                counter['carrier_idle_advance'] += 1
            if combat.contacts and combat.supporters == 0:
                counter['unsupported_contact'] += 1
        self.tail[snapshot.round_number].append(dict(tick=snapshot.tick,
            site=controller.route.site if controller.route else None,
            phase=controller.attack_plan.phase if controller.attack_plan else None,
            carrier=holder.position if holder else None,
            flanks=dict(controller.route_planner.flank_entries), actors=actors,
            rally={key: value for key, value in (controller.site_entry_state or {}).items()
                   if key in ('active', 'launched', 'wait', 'elapsed')}))


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    source = BEST_DIRECTORY / OPPONENT / 'attacker_plant_best.pt'
    state = load_checkpoint(source)
    analysis, signature, _ = load_frozen_analysis(BEST_DIRECTORY, OPPONENT, scenario)
    policy, _ = load_plant(source, scenario, OPPONENT, signature)
    policy.requires_grad_(False)
    rounds, diagnostics = [], []
    for seed in EVALUATION_SEEDS:
        review = EntryReview()
        records, _ = rollout(OPPONENT, scenario, analysis, policy, state['config'], seed,
                             preset_name=PRESET, training=False, tick_callback=review.observe)
        rounds.extend(records)
        for record in records:
            diagnostics.append(dict(seed=seed, round=record['round'], reason=record['reason'],
                counts=dict(review.counts[record['round']]), tail=list(review.tail[record['round']]),
                planted=record['planted'], terminal_tick=record['tick']))
        print('Evaluated frozen best:', seed, flush=True)
    result = dict(best_set=state['completed_sets'], schema=state['schema'], seeds=EVALUATION_SEEDS,
                  optimizer_updates=0, evaluation=summarize_plant(rounds), diagnostics=diagnostics)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, ensure_ascii=False, default=lambda value: value.item()), encoding='utf-8')
    print('Saved:', OUTPUT_FILE)


if __name__ == '__main__':
    main()
