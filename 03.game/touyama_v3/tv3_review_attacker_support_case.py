"""Reproduce one frozen failure, then compare teachers from the identical state."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OPPONENT = 'frc_v1'
BEST_DIRECTORY = HERE / 'data' / 'best'
OUTPUT_DIRECTORY = HERE / 'logs' / 'attacker_right_review'
CASE_SEED = 804000142
CASE_ROUND = 6
INTERVENTION_TICK = 55
MODES = ('frozen', 'legacy_teacher', 'support_teacher')
PRESET = 'Touyama Gaming'
TORCH_THREADS = 1

import json
import torch
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_plant import PlantDQN, PlantEncoder
from touyama_v3.tv3_train_attacker_plant import rollout
from touyama_v3.tv3_review_attacker_right import RightAudit, json_default, write_row


class CaseComplete(Exception):
    def __init__(self, record):
        self.record = record


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    source = load_sources(scenario, (OPPONENT,), BEST_DIRECTORY, BEST_DIRECTORY)[OPPONENT]
    config = load_checkpoint(source['paths']['plant'])['config']
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records = []
    reference_state = None
    for mode in MODES:
        trace = OUTPUT_DIRECTORY / ('support_case_'+mode+'_trace.jsonl')
        switched = False
        with trace.open('w',encoding='utf-8') as stream:
            audit = RightAudit(stream,CASE_SEED,source['plant'],scenario)
            def callback(controller):
                nonlocal switched, reference_state
                snap = controller.snapshot
                if snap.round_number == CASE_ROUND and snap.tick == INTERVENTION_TICK and not switched:
                    state = dict(allies=[dict(slot=a.slot,pos=a.position,hp=a.hp,facing=a.facing,blind=a.blind)
                                         for a in snap.allies], sightings=[(s.enemy_id,s.position,s.source) for s in snap.sightings],
                                 timer=snap.round_timer,route=controller.route.cells)
                    if reference_state is not None and state != reference_state:
                        raise RuntimeError('Pre-intervention public states differ')
                    reference_state = state
                    if mode != 'frozen':
                        if mode == 'support_teacher':
                            controller.policy = PlantDQN(11).eval().requires_grad_(False)
                            controller.encoder = PlantEncoder(scenario,11)
                        controller.training, controller.teacher_probability, controller.epsilon = True, 1., 0.
                        controller.cache = None
                        controller.prepare_team_tick()
                        audit.score_model = False
                    switched = True
                    write_row(stream,dict(event='intervention',mode=mode,seed=CASE_SEED,round=CASE_ROUND,
                                          tick=INTERVENTION_TICK,public_state=state,optimizer_updates=0))
                audit.observe(controller)
            def completed(record):
                if record['round'] == CASE_ROUND:
                    raise CaseComplete(record)
            try:
                rollout(OPPONENT,scenario,source['analysis'],source['plant'],config,CASE_SEED,
                        preset_name=PRESET,tick_callback=callback,round_callback=completed)
            except CaseComplete as completed_case:
                if not switched:
                    raise RuntimeError('Case ended before the configured intervention tick')
                records.append(dict(mode=mode,scope='same_state_teacher_intervention_not_trained_performance',
                    trace=str(trace),optimizer_updates=0,teacher_probability=0. if mode=='frozen' else 1.,
                    record=completed_case.record))
                print(mode,completed_case.record['reason'],completed_case.record['tick'],
                      completed_case.record['alive'],completed_case.record['remaining_hp'],flush=True)
    output=OUTPUT_DIRECTORY/'support_case_comparison.json'
    output.write_text(json.dumps(dict(source_hashes=source['hashes'],scenario=scenario.signature,
        seed=CASE_SEED,round=CASE_ROUND,intervention_tick=INTERVENTION_TICK,
        identical_pre_intervention_public_state=reference_state,runs=records),
        ensure_ascii=False,default=json_default,indent=2),encoding='utf-8')
    print('Saved:',output)


if __name__=='__main__':
    main()
