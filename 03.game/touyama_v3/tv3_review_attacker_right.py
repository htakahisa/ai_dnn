"""Frozen-model route/action audit. No learning or checkpoint writes."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Normal execution: cd touyama_v3; python tv3_review_attacker_right.py
OPPONENT = "frc_v1"
BEST_DIRECTORY = HERE / "data" / "best"
EVALUATION_SEEDS = (804000042, 804000142, 804000242)
PRESET = "Touyama Gaming"
OUTPUT_DIRECTORY = HERE / "logs" / "attacker_right_review"
OUTPUT_PREFIX = "baseline"
TORCH_THREADS = 1
PROBE_COLLECTION = False  # True evaluates the new route sampler using frozen models; never trains.

EVALUATION_MODE = 'frozen'  # frozen / legacy_teacher / support_teacher / exposure_teacher; no weight updates.

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from unittest.mock import patch

import numpy as np
import torch

from frc_v1.actions import MOVE_STEPS, build_masks, KINDS
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_train_attacker_plant import rollout, summarize_plant
from touyama_v3.tv3_learn_attacker_support import support_targets, support_path
from touyama_v3.tv3_learn_attacker_fire_support import shot_clear, public_threats
from grid_lines import wall_line_cells, line_cells


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def candidate_row(candidate):
    route = candidate['route']
    return dict(site=route.site, key=route.key, cells=route.cells,
                branches=route.branches, **{k: v for k, v in candidate.items() if k != 'route'})


class RightAudit:
    """Observe only actor-public inputs and own executed states; never select actions."""
    def __init__(self, stream, seed, policy, scenario, *, score_model=True):
        self.stream, self.seed, self.policy, self.scenario = stream, seed, policy, scenario
        self.controllers = set()
        self.counts = defaultdict(Counter)
        self.analysis_calls = {}
        self.score_model = score_model

    def analyze(self, original, encoder, snapshot, observation, candidates=None):
        result = original(encoder, snapshot, observation, candidates)
        if candidates is not None and len(candidates) > 1:
            self.analysis_calls[(snapshot.round_number, snapshot.tick)] = [
                candidate_row(c) for c in result['candidates']]
        return result

    def observe(self, controller):
        if id(controller) not in self.controllers:
            self.controllers.add(id(controller))
            original = controller.observe_executed_moves

            def executed(chars):
                original(chars)
                by_name = {str(getattr(c, 'base_name', c.name)): c for c in chars}
                if not controller.decisions:
                    return
                snap = controller.snapshot
                write_row(self.stream, dict(event='executed', seed=self.seed,
                    round=snap.round_number, tick=snap.tick,
                    actors=[dict(slot=before.slot, before=before.position,
                        after=tuple(map(int, by_name[name].pos)),
                        alive=bool(by_name[name].is_alive), hp=float(by_name[name].hp),
                        moved=tuple(by_name[name].pos) != before.position,
                        action=inputs.actions[selected].kind)
                        for name, (selected, inputs, before) in controller.decisions.items()]))

            controller.observe_executed_moves = executed
        snap = controller.snapshot
        if snap is None:
            return
        actors = []
        masks = build_masks(snap)
        targets = support_targets(self.scenario, snap, controller.route, controller.analysis_encoder.history.tracks)
        threats = public_threats(snap, controller.analysis_encoder.history.tracks)
        for name, (selected, inputs, ally) in controller.plans.items():
            if self.score_model:
                with torch.no_grad():
                    q = self.policy(torch.tensor(inputs.observation).unsqueeze(0))[0].numpy()
            else:
                q = np.zeros(len(inputs.mask))
            def score(index):
                return float(q[index]) if self.score_model else None
            legal = np.flatnonzero(inputs.mask)
            top = sorted(legal, key=lambda i: (-float(q[i]), int(i)))[:5]
            moves = []
            for kind_index, kind in enumerate(('STAY', 'N', 'E', 'S', 'W')):
                dr, dc = MOVE_STEPS.get(kind, (0, 0))
                pos = (ally.position[0]+dr, ally.position[1]+dc)
                indexes = [i for i in range(kind_index*8, kind_index*8+8) if inputs.mask[i]]
                inside = 0 <= pos[0] < self.scenario.grid.shape[0] and 0 <= pos[1] < self.scenario.grid.shape[1]
                moves.append(dict(kind=kind, position=pos, legal=bool(indexes),
                    engine_legal=bool(masks.kind[ally.slot, KINDS.index(kind)]),
                    q=max((float(q[i]) for i in indexes), default=None) if self.score_model else None,
                    unfiltered_q=max((float(q[i]) for i in range(kind_index*8,kind_index*8+8)
                                      if inputs.engine_movement_mask[i]), default=None) if self.score_model else None,
                    excluded_by=sorted(set(getattr(inputs, 'movement_debug', {}).get(kind, ()))) ,
                    goal_distance=int(inputs.distances[pos]) if inside else None,
                    traffic_distance=int(inputs.traffic[pos]) if inside else None,
                    site_distance=int(self.scenario.site_dist[controller.route.site][pos]) if inside else None))
            chosen = inputs.actions[selected]
            teacher = inputs.actions[inputs.teacher]
            counts = self.counts[snap.round_number]
            if ally.has_spike:
                counts['carrier_ticks'] += 1
                counts['carrier_'+chosen.kind] += 1
                counts['carrier_reason_'+inputs.combat.reason] += 1
                if chosen.kind == 'STAY' and teacher.kind in MOVE_STEPS:
                    counts['carrier_ignored_move_teacher'] += 1
                if chosen.kind == 'STAY' and inputs.combat.reason == 'advance':
                    counts['carrier_idle_advance'] += 1
                if inputs.mask[-2] and chosen.kind != 'PLANT':
                    counts['carrier_ignored_legal_plant'] += 1
            actors.append(dict(name=name, slot=ally.slot, position=ally.position,
                facing=ally.facing, hp=ally.hp, blind=ally.blind,
                movement_disabled=ally.movement_disabled, carrier=ally.has_spike,
                plant_progress=ally.plant_progress, goal=inputs.goal,
                cursor=controller.cursors[ally.slot], action=chosen.kind,
                action_index=selected, action_facing=chosen.facing, target=chosen.target,
                q=score(selected), teacher=teacher.kind,
                teacher_index=inputs.teacher, teacher_q=score(inputs.teacher),
                combat=inputs.combat.reason, contacts=inputs.combat.contacts,
                supporters=inputs.combat.supporters, moves=moves,
                support_path=support_path(self.scenario,snap,ally,targets,controller.route),
                shot_geometry=[dict(target=s.position, source=s.source,
                    wall_cells=[p for p in wall_line_cells(ally.position,s.position) if self.scenario.grid[p] == 1],
                    ally_blockers=[a.slot for a in snap.allies if a.alive and a.slot != ally.slot
                                   and a.position in line_cells(ally.position,s.position)[1:-1]],
                    shot_clear=shot_clear(self.scenario,snap,ally.position,s.position,ally.slot)) for s in threats],
                top_actions=[dict(index=int(i),kind=inputs.actions[i].kind,
                    facing=inputs.actions[i].facing,q=score(i)) for i in top] if self.score_model else []))
        write_row(self.stream, dict(event='decision', seed=self.seed, round=snap.round_number,
            tick=snap.tick, timer=snap.round_timer,
            site=controller.route.site if controller.route else None,
            route=controller.route.cells if controller.route else None,
            phase=controller.attack_plan.phase if controller.attack_plan else None,
            plan_reason=controller.attack_plan.reason if controller.attack_plan else None,
            planner_progress_tick=controller.route_planner.progress_tick,
            route_candidates=self.analysis_calls.pop((snap.round_number, snap.tick), None),
            route_decision=controller.route_planner.events[-1] if controller.route_planner.events else None,
            rally=controller.site_entry_state, assembly=dict(assigned=controller.site_assembly.assigned,
                entries=controller.site_assembly.entries, entered=controller.site_assembly.entered),
            sightings=[dict(enemy_id=s.enemy_id,position=s.position,source=s.source) for s in snap.sightings],
            actors=actors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=EVALUATION_SEEDS)
    parser.add_argument('--prefix', default=OUTPUT_PREFIX)
    parser.add_argument('--mode', choices=('frozen','legacy_teacher','support_teacher','exposure_teacher','corner_teacher'), default=EVALUATION_MODE)
    parser.add_argument('--probe-collection', action=argparse.BooleanOptionalAction, default=PROBE_COLLECTION)
    args = parser.parse_args()
    if args.probe_collection and args.mode != 'frozen':
        raise ValueError('Route exploration and teacher diagnostics must be evaluated separately')
    if Path(args.prefix).name != args.prefix:
        raise ValueError('prefix must be a filename component')
    torch.set_num_threads(TORCH_THREADS)
    scenario = Scenario()
    source = load_sources(scenario, (OPPONENT,), BEST_DIRECTORY, BEST_DIRECTORY)[OPPONENT]
    state = load_checkpoint(source['paths']['plant'])
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    policy = source['plant']
    teacher_mode = args.mode != 'frozen'
    if args.mode in ('support_teacher', 'exposure_teacher', 'corner_teacher'):
        from touyama_v3.tv3_learn_attacker_plant import PlantDQN
        policy = PlantDQN(12 if args.mode=='corner_teacher' else 11).eval().requires_grad_(False)
    all_rounds, diagnostics = [], []
    original = source['analysis'].analyze
    trace = OUTPUT_DIRECTORY / (args.prefix + '_trace.jsonl')
    with trace.open('w', encoding='utf-8') as stream:
        for seed in args.seeds:
            audit = RightAudit(stream, seed, policy, scenario, score_model=not teacher_mode)
            experimental_controller = None
            def starting(controller, round_number):
                nonlocal experimental_controller
                if args.mode == 'exposure_teacher':
                    from touyama_v3.tv3_review_attacker_exposure_teacher import install_probe
                    experimental_controller = controller
                    install_probe(controller)
            def completed(record):
                nonlocal experimental_controller
                if experimental_controller is not None:
                    # Restore before reset_round rebuilds combat state; no probe state leaks across rounds.
                    experimental_controller.encoder = experimental_controller.encoder.base
                    experimental_controller = None
            def analyze(*values, **kwargs):
                return audit.analyze(original, *values, **kwargs)
            with patch.object(source['analysis'], 'analyze', side_effect=analyze):
                if args.probe_collection:
                    from touyama_v3.tv3_collect_attacker_analysis import play_block
                    from touyama_v3.tv3_train_attacker_analysis import (
                        EXPLORATION_RATE, EXPLORATION_SCOPE, EXPLORATION_PROBE_STEPS, EXPLORATION_PROBE_TICKS,
                        EXPLORATION_ROUND_START_ONLY)
                    config = {**state['config'], 'collection_executor': 'fixed_plant',
                        'exploration_scope': EXPLORATION_SCOPE,
                        'exploration_probe_steps': EXPLORATION_PROBE_STEPS,
                        'exploration_probe_ticks': EXPLORATION_PROBE_TICKS,
                        'exploration_round_start_only': EXPLORATION_ROUND_START_ONLY}
                    records, _ = play_block(OPPONENT, scenario, source['analysis'], config, seed,
                        EXPLORATION_RATE, preset_name=PRESET, plant_policy=source['plant'],
                        tick_callback=audit.observe)
                else:
                    records, _ = rollout(OPPONENT, scenario, source['analysis'], policy,
                        state['config'], seed, preset_name=PRESET, training=teacher_mode,
                        teacher_probability=1. if teacher_mode else 0.,
                        tick_callback=audit.observe, round_callback=completed, round_start_callback=starting)
            all_rounds.extend(dict(seed=seed, **r) for r in records)
            diagnostics.extend(dict(seed=seed, round=r['round'], counts=dict(audit.counts[r['round']])) for r in records)
            print('Evaluation completed:', args.mode, seed, flush=True)
    if args.probe_collection:
        from touyama_v3.tv3_train_attacker_analysis import summarize
        evaluation = {**summarize(all_rounds), 'by_site': {side: summarize([
            r for r in all_rounds if r['target_site'] == side]) for side in ('L', 'R')}}
    else:
        evaluation = summarize_plant(all_rounds)
    result = dict(opponent=OPPONENT, source_hashes=source['hashes'],
        scenario=scenario.signature, plant_version=policy.execution_version, source_plant_version=state['schema']['version'],
        best_set=state['completed_sets'] if not teacher_mode else None, optimizer_updates=0,
        evaluation_mode=args.mode, teacher_probability=1. if teacher_mode else 0.,
        epsilon=0, seeds=args.seeds, trace=str(trace), evaluation=evaluation,
        route_exploration=EXPLORATION_RATE if args.probe_collection else 0.,
        scope='collection_sampler_diagnostic' if args.probe_collection else 'teacher_diagnostic_not_learned_performance' if teacher_mode else 'production_plant_evaluation',
        experimental_exposure=args.mode == 'exposure_teacher',
        experimental_flash_window_teacher=getattr(policy,'experimental_flash_teacher',False),
        exposure_probe_hash=hashlib.sha256((HERE / 'tv3_review_attacker_exposure_teacher.py').read_bytes()).hexdigest()
            if args.mode == 'exposure_teacher' else None,
        rounds=all_rounds, diagnostics=diagnostics,
        script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = OUTPUT_DIRECTORY / (args.prefix + '_summary.json')
    output.write_text(json.dumps(result, ensure_ascii=False, default=json_default), encoding='utf-8')
    print('Saved:', output)
    print('Outcomes:', dict(Counter(r['reason'] for r in all_rounds)))


if __name__ == '__main__':
    main()
