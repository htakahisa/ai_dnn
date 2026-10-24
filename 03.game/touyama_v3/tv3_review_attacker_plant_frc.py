"""Read-only plant evaluation and public action diagnostics; no optimizer updates."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OPPONENT = "frc_v1"
BEST_DIRECTORY = HERE / "data" / "best"
OUTPUT_DIRECTORY = HERE / "logs" / "attacker_plant_review"
TORCH_THREADS = 1
EVALUATION_SEED_LIMIT = 0
OUTPUT_FILE = "frc_v1_current_execution.json"  # Preserve the original baseline report.
ALLOW_VERSION5_DIAGNOSTIC = True  # Read-only zero extension, never save or deploy.
ALLOW_OLD_LOS_DIAGNOSTIC = True  # Same weights under revised geometry, never a production loader.

import io
import json
import hashlib
from collections import Counter, defaultdict
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_plant import load_plant, load_frozen_analysis, PlantDQN, policy_schema
from touyama_v3.tv3_train_attacker_plant import rollout, summarize_plant
from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel


class PublicActionReview:
    def __init__(self):
        self.counts = Counter()
        self.rounds = defaultdict(Counter)
        self.examples = []

    def observe(self, controller):
        snapshot = controller.snapshot
        for name, (selected, inputs, ally) in controller.plans.items():
            kind, teacher = inputs.actions[selected].kind, inputs.actions[inputs.teacher].kind
            from frc_v1.actions import MOVE_STEPS
            advancing = sum(inputs.mask[i] and inputs.actions[i].kind in MOVE_STEPS
                and 0 <= inputs.distances[(ally.position[0]+MOVE_STEPS[inputs.actions[i].kind][0],
                                           ally.position[1]+MOVE_STEPS[inputs.actions[i].kind][1])] < inputs.distances[ally.position]
                for i in range(8,40))
            counts = self.rounds[snapshot.round_number]
            for counter in (self.counts, counts):
                counter[kind] += 1
                counter['teacher_'+teacher] += 1
                if kind == 'STAY' and teacher in ('N','S','E','W'):
                    counter['teacher_move_ignored'] += 1
                if inputs.mask[-2] and kind != 'PLANT':
                    counter['plant_legal_ignored'] += 1
                if ally.has_spike and kind == 'STAY' and inputs.distances[ally.position] > 0 and not inputs.combat.contacts:
                    counter['quiet_carrier_stall'] += 1
                    counter['carrier_stall_reason_'+inputs.combat.reason] += 1
                    if advancing:
                        counter['carrier_stall_with_legal_progress'] += 1
                    else:
                        counter['carrier_stall_without_legal_progress'] += 1
            if len(self.examples) < 15 and ally.has_spike and kind == 'STAY' and inputs.distances[ally.position] > 0:
                self.examples.append({'round':snapshot.round_number,'tick':snapshot.tick,'name':name,
                    'position':ally.position,'goal':inputs.goal,'distance':int(inputs.distances[ally.position]),
                    'teacher':teacher,'legal_moves':int(inputs.mask[8:40].sum()),
                    'mode':controller.route_mode,'phase':controller.attack_plan.phase if controller.attack_plan else None,
                    'combat_reason':inputs.combat.reason,'legal_progress':int(advancing),
                    'edge_counts':[(list(edge),count) for (slot,edge),count in controller.edges.items()
                                   if slot==ally.slot and ally.position in edge and count>=2]})


def diagnostic_policy(content, state, scenario, signature):
    if state['schema']['version'] != 5:
        return load_plant(io.BytesIO(content), scenario, OPPONENT, signature)[0], False
    if not ALLOW_VERSION5_DIAGNOSTIC:
        raise ValueError('Version5 diagnostic adaptation is disabled')
    expected = policy_schema(scenario)
    expected.update(version=5, obs_dim=expected['obs_dim']-2, execution='adaptive_combat_plant_v5')
    expected.pop('edge_window')
    expected.pop('plant_legal_progress_features')
    expected.pop('wall_los')
    if ALLOW_OLD_LOS_DIAGNOSTIC:
        expected['scenario'] = state['schema']['scenario']
    if (json.dumps(expected,sort_keys=True) != json.dumps(state['schema'],sort_keys=True)
            or state['opponent'] != OPPONENT or state['analysis_hash'] != signature):
        raise ValueError('Old diagnostic model conditions differ')
    model = PlantDQN()
    weights = dict(state['model'])
    old = weights['net.0.weight']
    extended = torch.zeros((old.shape[0],old.shape[1]+2),dtype=old.dtype)
    extended[:,:old.shape[1]] = old
    weights['net.0.weight'] = extended
    model.load_state_dict(weights)
    model.eval()
    model.requires_grad_(False)
    return model, True


def diagnostic_analysis(directory, scenario):
    content = (directory/OPPONENT/'attacker_analysis_best.pt').read_bytes()
    state = torch.load(io.BytesIO(content),map_location='cpu',weights_only=True)
    expected = AttackerEncoder(scenario).schema()
    if not ALLOW_OLD_LOS_DIAGNOSTIC or expected['scenario'] == state['schema']['scenario']:
        return load_frozen_analysis(directory,OPPONENT,scenario)[:2]
    expected['scenario'] = state['schema']['scenario']
    if (json.dumps(expected,sort_keys=True) != json.dumps(state['schema'],sort_keys=True)
            or state['opponent'] != OPPONENT):
        raise ValueError('Old analysis diagnostics differ beyond LOS scenario')
    encoder = AttackerEncoder(scenario)
    model = AttackerAnalysisModel(len(encoder.fields),len(encoder.route_fields),len(scenario.names))
    model.load_state_dict(state['model'])
    model.trained_rounds = state['trained_rounds']
    model.eval()
    model.requires_grad_(False)
    return model,hashlib.sha256(content).hexdigest()


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    source = BEST_DIRECTORY / OPPONENT / 'attacker_plant_best.pt'
    content = source.read_bytes()
    state = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
    analysis, signature = diagnostic_analysis(BEST_DIRECTORY, scenario)
    policy, adapted = diagnostic_policy(content, state, scenario, signature)
    review = PublicActionReview()
    records = []
    per_seed = []
    plan = state['evaluation']['roster_plan']
    for row in plan[:EVALUATION_SEED_LIMIT] if EVALUATION_SEED_LIMIT else plan:
        records_for_seed, _ = rollout(OPPONENT, scenario, analysis, policy, state['config'], row['seed'],
            preset_name=row['preset'], training=False, tick_callback=review.observe)
        records.extend(records_for_seed)
        per_seed.append({'seed':row['seed'], 'rounds':len(records_for_seed)})
        print('Evaluated seed', row['seed'], flush=True)
    result = {'opponent':OPPONENT,'best_set':state['completed_sets'],
        'source_hash':hashlib.sha256(content).hexdigest(),'analysis_hash':signature,
        'teacher_probability':0,'epsilon':0,'optimizer_updates':0,
        'version5_zero_extension':adapted,'execution_schema':policy_schema(scenario),
        'source_scenario':state['schema']['scenario'],'execution_scenario':scenario.signature,
        'evaluation':summarize_plant(records),'end_reasons':dict(Counter(r['reason'] for r in records)),
        'action_counts':dict(review.counts),'public_examples':review.examples,
        'records':records,'seeds':per_seed}
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIRECTORY / OUTPUT_FILE
    path.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    print('Saved read-only diagnostics:',path)
    print(json.dumps({k:result[k] for k in ('best_set','end_reasons','action_counts')},ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
