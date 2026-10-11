"""Train opponent-specific attacker plant policies with frozen analysis routes."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 通常設定：このファイル冒頭を編集し、touyama_v3からオプションなしで実行。
TRAINING_SETS = 5  # FRCのv19教師修正の検証学習。1セット=12ラウンド。
TOTAL_TRAINING_SET_LIMIT = 5  # set5で自動停止してから実力評価を確認する。
MAX_PARALLEL_WORKERS = 6  # 相手AIごとの最大同時実行数。1で順次実行。
RESUME_TRAINING = False
WARM_START_FROM_BEST = True  # Reuse compatible weights; start a fresh optimizer and replay.

ALL_TARGET_OPPONENTS = ("concon_v1", "omoko_v1", "fnatic_v3", "frc_v1", "toru_ai_v4", "ghost_champions_v2")
TARGET_OPPONENTS = ("frc_v1",)  # 検証後に全対象へ広げる場合はALL_TARGET_OPPONENTSへ変更。
#TARGET_OPPONENTS = ("frc_v1",)  # 1種類でも末尾のカンマが必要。
TRAINING_PRESETS = ("Touyama Gaming",)
EVALUATION_PRESETS = ("Touyama Gaming",)  # 同じ5人・独立seedで評価。
EVALUATION_ONLY = False
RANDOM_SEED = 42
EVALUATION_INTERVAL = 5
EVALUATION_SEED_COUNT = 3
TRAINING_START_ROUNDS = (1, 13)
EVALUATION_START_ROUNDS = (1, 13)
BEST_SELECTION_POLICY = 'man_advantaged_plant_then_competitive_plant_then_ability_advantage_v1'
BEST_SELECTION_PRIORITIES = ('advantaged_plant_rate','competitive_plant_rate','ability_advantage_plant_rate')
SAVE_EVALUATION_CHECKPOINTS = True  # Keep evaluated weights even when they are not selected as best.
SAVE_REVIEW_WEIGHT_CHECKPOINTS = True  # 学習前・各セットの重みだけを実行ごとに保存し、劣化箇所を後から比較する。
OPTIMIZER_UPDATES = 200
BATCH_SIZE = 64
REPLAY_SIZE = 60000
LEARNING_RATE = .001
EPSILON_START = .20
EPSILON_END = .03
TEACHER_START = .95
TEACHER_END = .10
EXPLORATION_DECAY_SETS = 60
ROUTE_EXPLORATION_START = .30  # Separate from local move/ability epsilon.
ROUTE_EXPLORATION_END = .05
ROUTE_EXPLORATION_PROBE_STEPS = 24
ROUTE_EXPLORATION_PROBE_TICKS = 40
DEMONSTRATION_WEIGHT = .05
DEMONSTRATION_KIND_WEIGHT = .25
CARRIER_SAMPLE_FRACTION = .50
FRC_DEMONSTRATION_WEIGHT = .15
FRC_DEMONSTRATION_KIND_WEIGHT = 1.
DEMONSTRATION_TEACHER_VARIANT = True  # 種類だけでなく、教師の向き・投擲先へ正の勾配を入れる。
FRC_CARRIER_SAMPLE_FRACTION = .65
FRC_OPTIMIZER_UPDATE_MULTIPLIER = 2
TORCH_THREADS = 1
MAX_ROUND_STEPS = 400
ANALYSIS_DIRECTORY = HERE / "data" / "best"
DATA_DIRECTORY = HERE / "data" / "attacker_plant"
BEST_DIRECTORY = HERE / "data" / "best"
LOG_DIRECTORY = HERE / "logs" / "attacker_plant"
COLOR_CONSOLE = True
EVALUATION_CONSOLE_COLOR = "\033[1;32m"

# プラント成功を主目的とする。facing補助は敵との初接触時だけ。
PLANT_SUCCESS_REWARD = 8.
PLANT_COMPLETION_REWARD = 1.  # Every real plant is progress, including an outnumbered plant.
PLANT_FAILURE_REWARD = -6.
PLANT_TIMEOUT_REWARD = -10.  # Deliberate failure must not become cheap by waiting.
FAILURE_DISCOUNT_COMPENSATION = True
SURVIVOR_REWARD = 0.  # Survival is already represented by the team player ratio.
RESOURCE_REWARD = .10
TICK_PENALTY = .01
PROGRESS_REWARD = .04
DAMAGE_PENALTY = .01
DEATH_PENALTY = 1.
KILL_REWARD = .25
TEAM_DAMAGE_REWARD = .002
ABILITY_USE_PENALTY = .02
PLANT_PROGRESS_REWARD = .20
PREAIM_REWARD = .04
NEW_INFORMATION_REWARD = .03
STATIONARY_HIT_REWARD = .04
UTILITY_EFFECT_REWARD = .10
UTILITY_NO_EFFECT_PENALTY = .05
GROUP_DISTANCE_LIMIT = 4
GROUP_SEPARATION_PENALTY = .01
LOW_DAMAGE_MIN_SURVIVORS = 5
LOW_DAMAGE_HP_FRACTION = .90
LOW_DAMAGE_PLANT_REWARD = 0.  # Keep the statistic, not a pristine-HP objective.
PLANT_HP_RETENTION_REWARD = 0.  # Avoid double-counting pristine HP at completion.
COOPERATIVE_HIT_REWARD = .08
SHARED_FIRE_PROGRESS_REWARD = .04
CARRIER_STALL_PENALTY = .08
SUPPORT_PROGRESS_REWARD = .08  # v11: discounted public firing-lane distance potential.

import argparse
from collections import deque, Counter
import contextlib
import copy
import json
import logging
import os
import uuid

import numpy as np
import torch

from game_core import FACING_VECTORS
from touyama_v3.tv3_scenario import Scenario, OPPONENTS
from touyama_v3.tv3_train_defender_analysis import legacy_root, seed_all, relocate_debug_logs
from touyama_v3.tv3_train_attacker_analysis import (
    summarize, format_summary, format_last_evaluation, atomic_save, ConsoleFormatter, console_supports_color,
)
from touyama_v3.tv3_learn_attacker_plant import (
    PlantDQN, OBS_DIM, ACTION_DIM, policy_schema, load_frozen_analysis, learn_plant,
)
from touyama_v3.tv3_learn_attacker_analysis import branch_sequence
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
from touyama_v3.tv3_attacker_site_entry import mission_progress
from touyama_v3.tv3_attacker_rosters import eligible_attacker_presets
from touyama_v3.tv3_checkpoint import load_checkpoint


def reward_config():
    return {key: value for key, value in globals().items() if key.endswith("_REWARD") or key.endswith("_PENALTY")}


def warm_start_source(path,schema,analysis_hash,opponent,directory):
    """Archive exact compatible input weights; never restore old reward replay."""
    import io,hashlib
    payload=Path(path).read_bytes()
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    if not warm_start_schema_compatible(state['schema'],schema) or state['analysis_hash']!=analysis_hash or state['opponent']!=opponent:
        return None,None
    digest=hashlib.sha256(payload).hexdigest()
    destination=Path(directory)/opponent/f'{digest}.pt'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
            raise ValueError('Warm-start archive hash mismatch')
    else:
        temporary=destination.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(destination)
    return state,digest


def warm_start_schema_compatible(source,target):
    if source==target:
        return True
    if source.get('version')==16 and target.get('version')==18:
        omitted={'version','execution','entry_yield','deadline_route_reserve'}
        return {k:v for k,v in source.items() if k not in omitted}=={k:v for k,v in target.items() if k not in omitted}
    if source.get('version') in (16,18) and target.get('version')==19:
        omitted={'version','execution','entry_yield','deadline_route_reserve',
                 'masked_retreat_teacher','masked_teacher_facing'} if source['version']==16 else {
                 'version','execution','masked_retreat_teacher','masked_teacher_facing'}
        return {k:v for k,v in source.items() if k not in omitted}=={k:v for k,v in target.items() if k not in omitted}
    if source.get('version')!=13 or target.get('version') not in (14,16):
        return False
    # Initialization only: no optimizer/replay migration or checkpoint retagging.
    omitted={'version','execution','recent_edge_progress_exception','empty_evidence','temporary_body_cover',
             'flash_progress_release','flash_step_teacher','flash_rally_release'}
    return {k:v for k,v in source.items() if k not in omitted}=={k:v for k,v in target.items() if k not in omitted}


def carrier_stall_cost(before, inputs, action):
    """Penalize avoidable idle ticks, preserving combat and coordinated waits."""
    if (not before.has_spike or action.kind != 'STAY' or before.blind
            or before.plant_progress or inputs.combat is None
            or inputs.combat.contacts or inputs.combat.reason != 'advance'
            or inputs.actions[inputs.teacher].kind == 'STAY'):
        return 0.
    return CARRIER_STALL_PENALTY if inputs.observation[-2:].max() > .5 else 0.


def support_progress_credit(before, after, inputs, done):
    field = inputs.support_distances
    if field is None:
        return 0.
    from touyama_v3.tv3_learn_attacker_plant import GAMMA
    from touyama_v3.tv3_learn_attacker_support import SUPPORT_SEARCH_STEPS
    def potential(position):
        distance = field[tuple(position)]
        return -min(SUPPORT_SEARCH_STEPS, distance if distance >= 0 else SUPPORT_SEARCH_STEPS) / SUPPORT_SEARCH_STEPS
    return SUPPORT_PROGRESS_REWARD * ((0. if done else GAMMA * potential(after)) - potential(before.position))


def learning_settings(opponent, updates):
    frc = opponent == 'frc_v1'
    return {'demonstration_weight': FRC_DEMONSTRATION_WEIGHT if frc else DEMONSTRATION_WEIGHT,
        'demonstration_teacher_variant': DEMONSTRATION_TEACHER_VARIANT,
            'demonstration_kind_weight': FRC_DEMONSTRATION_KIND_WEIGHT if frc else DEMONSTRATION_KIND_WEIGHT,
            'carrier_sample_fraction': FRC_CARRIER_SAMPLE_FRACTION if frc else CARRIER_SAMPLE_FRACTION,
            'updates': updates * (FRC_OPTIMIZER_UPDATE_MULTIPLIER if frc else 1)}


def group_separation_cost(scenario, snapshot, position, before, combat, plan, has_flanks, *, damage=0., terminal=False):
    """Use the executed tick's plan; missing plans never incur formation cost."""
    if (terminal or snapshot is None or plan is None or plan.phase == "entry" or has_flanks
            or before.blind or damage > 0 or combat is None or combat.contacts > 0):
        return 0.
    holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
    if holder is None:
        return 0.
    from grid_paths import distance_map
    separation = distance_map(scenario.grid, holder.position)[tuple(position)]
    return GROUP_SEPARATION_PENALTY * min(6, max(0, separation - GROUP_DISTANCE_LIMIT))


def first_contact_preaim(scenario, origin, target, direction):
    """Training-only correctness check; hidden targets never enter action inputs."""
    if origin == target or not scenario.clear(origin, target):
        return 0.
    vector = np.asarray((target[1] - origin[1], target[0] - origin[0]), dtype=float)
    facing_vector = np.asarray(FACING_VECTORS[direction], dtype=float)
    similarity = float(vector @ facing_vector / (np.linalg.norm(vector) * np.linalg.norm(facing_vector)))
    return PREAIM_REWARD if similarity >= .70710678 else 0.


def utility_effect_credit(record):
    observed = min(3, len(record["affected_enemies"]) + int(record["blocked_lines"] > 0))
    credit = UTILITY_EFFECT_REWARD*observed if observed else -UTILITY_NO_EFFECT_PENALTY
    if record["ability"] == "RECON":
        credit += NEW_INFORMATION_REWARD*len(record["affected_enemies"])
    return credit


def terminal_succeeded(terminal):
    if 'preplant_round_won' in terminal:
        return terminal['planted'] or terminal['preplant_round_won']
    if terminal.get('reason')=='defender_eliminated' and terminal.get('enemy_alive')==0 and terminal['alive']>0:
        return True
    return terminal['planted']


def terminal_bonus(terminal, initial_alive, reserve):
    if not terminal_succeeded(terminal):
        return PLANT_TIMEOUT_REWARD if terminal.get('reason')=='timeout' else PLANT_FAILURE_REWARD
    hp_fraction = terminal.get('remaining_hp', 0.) / max(1., terminal.get('initial_team_hp', 1.))
    low_damage = (terminal['alive'] >= LOW_DAMAGE_MIN_SURVIVORS and hp_fraction >= LOW_DAMAGE_HP_FRACTION)
    total_players=terminal['alive']+terminal['enemy_alive'] if 'enemy_alive' in terminal else initial_alive
    if 'remaining_abilities_alive' in terminal and 'enemy_remaining_abilities_alive' in terminal:
        own_charges=terminal['remaining_abilities_alive']
        reserve=own_charges/max(1,own_charges+terminal['enemy_remaining_abilities_alive'])
    return (PLANT_COMPLETION_REWARD + PLANT_SUCCESS_REWARD * terminal["alive"] / max(1,total_players)
            + SURVIVOR_REWARD * terminal["alive"] + RESOURCE_REWARD * reserve
            + PLANT_HP_RETENTION_REWARD * min(1., hp_fraction) + LOW_DAMAGE_PLANT_REWARD * low_damage
            if terminal_succeeded(terminal) else PLANT_FAILURE_REWARD)


def apply_terminal_bonus(transitions, indices, alive, terminal, initial_alive, reserve, decision_counts=None):
    bonus = terminal_bonus(terminal, initial_alive, reserve)
    total = 0.
    for n, index in indices.items():
        if terminal_succeeded(terminal) and not alive[n]:
            continue
        outcome_bonus=bonus
        if not terminal_succeeded(terminal):
            # A dead actor cannot choose to keep waiting after their own death.
            if not alive[n]:
                outcome_bonus=PLANT_FAILURE_REWARD
            if FAILURE_DISCOUNT_COMPENSATION:
                from touyama_v3.tv3_learn_attacker_plant import GAMMA
                count=(decision_counts or {}).get(n,1)
                if count<1:
                    raise ValueError('A terminal actor must have at least one recorded decision')
                outcome_bonus /= GAMMA**(count-1)
        transitions[index][2] += outcome_bonus
        transitions[index][5] = 1.
        total += outcome_bonus
    return total


def rollout(opponent, scenario, analysis, policy, config, seed, *, preset_name,
            training=False, epsilon=0., teacher_probability=0., tick_callback=None, route_exploration=0.,
            round_callback=None, round_start_callback=None, start_round=1):
    if not isinstance(start_round, int) or start_round < 1:
        raise ValueError('start_round must be a positive match round number')
    rounds, transitions, history = [], [], []
    from touyama_v3.tv3_attacker_combat_audit import CombatAudit
    from simulation_runtime import cpu_inference
    with legacy_root(), open(os.devnull, "w", encoding="utf-8") as quiet, \
            contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet), cpu_inference(enabled=True):
        from party_presets import get_preset
        from run_game import VisualFPSBattle
        from touyama_v3.tv3_opponents import build_opponent_ai as _build_team_ai
        from team_ai import DualRoleTeamAI
        from controllers import DefaultDefenderController
        seed_all(seed)
        own_preset = get_preset(preset_name)
        enemy_preset = get_preset(OPPONENTS[opponent][1])
        from touyama_v3.tv3_opponents import scoped_presets
        own_preset, enemy_preset = scoped_presets(own_preset, enemy_preset)
        controller = TouyamaV3AttackerPlantController(scenario, analysis, policy, seed=seed + 19, training=training)
        controller.epsilon, controller.teacher_probability = epsilon, teacher_probability
        if training and route_exploration:
            controller.planning_rng = controller.rng
            controller.planning_exploration = route_exploration
            controller.planning_exploration_scope = 'all_routes'
            controller.planning_probe_steps = config['route_exploration']['probe_steps']
            controller.planning_probe_ticks = config['route_exploration']['probe_ticks']
            controller.planning_round_start_only = True
        team = DualRoleTeamAI("Touyama v3 plant trainer", lambda: controller, DefaultDefenderController)
        game = VisualFPSBattle(scenario.maze, team, _build_team_ai(OPPONENTS[opponent][0], device="cpu"), headless=True,
            attacker_roster=list(own_preset.players), defender_roster=list(enemy_preset.players),
            spike_holder_name=own_preset.spike_holder, defender_spike_holder_name=enemy_preset.spike_holder,
            attacker_igl_name=own_preset.igl, defender_igl_name=enemy_preset.igl,
            attacker_team_name=own_preset.name, defender_team_name=enemy_preset.name, disable_side_swap=True)
        game.stop_after_round, game.analytics_tracker = True, None
        game._record_replay_frame = lambda: None
        relocate_debug_logs(game.defender_controller, None)
        if start_round != 1:
            game.current_round = start_round
            game.init_round()
        for round_number in range(start_round, start_round + 12):
            if round_start_callback is not None:
                round_start_callback(controller, round_number)
            audit = CombatAudit() if training or config.get("combat_diagnostics", True) else None
            controller.previous_rounds = list(history)
            own = [c for c in game.chars if c.team == "A"]
            enemy = [c for c in game.chars if c.team == "D"]
            initial_alive, initial_enemy_alive = sum(c.is_alive for c in own), sum(c.is_alive for c in enemy)
            initial_team_hp = float(sum(c.max_hp for c in own if c.is_alive))
            charges = {str(getattr(c, "base_name", c.name)): int(getattr(c, c.ability_name.lower() + "_charges", 0)) for c in own}
            initial_abilities = sum(charges.values())
            terminal = None
            damage = uses = reward_sum = 0.
            steps = 0
            last_indices = {}
            decision_counts = Counter()
            utility_indices = {}
            contacts = set()
            entry_counts = Counter()
            entry_launched = False
            score_before = game.attacker_wins
            paths = {str(getattr(c, 'base_name', c.name)): [tuple(map(int, c.pos))] for c in own}
            while not game.round_over and not game.match_over:
                closed = terminal is not None
                controller.decisions = {}
                if not game.defender_setup_phase.active and not closed:
                    controller.prepare_team_tick()  # Freeze public inputs before reading reward labels.
                    if audit and controller.snapshot:
                        audit.before(controller.snapshot, controller.actions, controller.analysis_encoder.history.tracks)
                decision_tick = controller.snapshot.tick if controller.snapshot else -1
                decision_plan = controller.attack_plan
                entry_state = controller.site_entry_state
                decision_has_flanks = bool(controller.route_planner.flank_entries or
                                           entry_state and entry_state['flank'])
                if not closed and not game.defender_setup_phase.active and entry_state:
                    entry_counts['rally_ticks'] += int(not entry_state['launched'])
                    if entry_state['launched'] and not entry_launched:
                        entry_counts['launches'] += 1
                    entry_launched = entry_state['launched']
                decision_snapshot = controller.snapshot
                if tick_callback is not None and not closed and not game.defender_setup_phase.active:
                    tick_callback(controller)
                previous_hp = {str(getattr(c, "base_name", c.name)): max(0., c.hp) if c.is_alive else 0. for c in own}
                previous_kills = {str(getattr(c, "base_name", c.name)): c.round_kills for c in own}
                previous_enemy_hp = sum(max(0., c.hp) for c in enemy if c.is_alive)
                # These exact cells are available only to the training reward evaluator.
                enemy_before = {i: tuple(map(int, c.pos)) for i, c in enumerate(enemy) if c.is_alive}
                game.step_tick()
                if not closed:
                    for c in own:
                        path = paths[str(getattr(c, 'base_name', c.name))]
                        pos = tuple(map(int, c.pos))
                        if pos != path[-1]:
                            path.append(pos)
                if audit:
                    audit.after(game)
                steps += 1
                if steps > config["max_round_steps"]:
                    raise RuntimeError(f"{opponent} round {round_number}: step limit exceeded")
                if closed:
                    continue  # Preserve opponent history through the real round; no guard rewards/transitions.
                current = {str(getattr(c, "base_name", c.name)): c for c in own}
                previous_charges = charges
                charges = {name: int(getattr(c, c.ability_name.lower() + "_charges", 0)) for name, c in current.items()}
                uses += sum(max(0, previous_charges[n] - charges[n]) for n in current)
                damage += sum(max(0., previous_hp[n] - (max(0., c.hp) if c.is_alive else 0.)) for n, c in current.items())
                enemy_damage = max(0., previous_enemy_hp - sum(max(0., c.hp) for c in enemy if c.is_alive))
                done_team = bool(game.is_planted or game.round_over or game.match_over)
                decisions = dict(controller.decisions)
                controller.observe_executed_moves(own)
                if not done_team:
                    controller.prepare_team_tick()
                    next_snapshot = controller.snapshot
                else:
                    next_snapshot = controller.sensor.build(game)
                sightings = next_snapshot.sightings if next_snapshot else ()
                for name, (action_index, inputs, before) in decisions.items():
                    char = current[name]
                    progress = mission_progress(scenario, before.position, char.pos)
                    own_damage = max(0., previous_hp[name] - (max(0., char.hp) if char.is_alive else 0.))
                    chosen_action = inputs.actions[action_index]
                    if before.has_spike and inputs.combat and inputs.combat.reason == 'advance':
                        entry_counts['carrier_blocked_ticks'] += int(
                            chosen_action.kind == 'STAY' and inputs.distances[before.position] > 0
                            and inputs.observation[-1] < .5)
                    if chosen_action.kind in ('N', 'E', 'S', 'W'):
                        static_progress = inputs.distances[before.position] - inputs.distances[tuple(char.pos)]
                        entry_counts['detour_steps'] += int(progress > 0 and static_progress <= 0)
                        if inputs.combat:
                            entry_counts['unsupported_contact_moves'] += int(
                                inputs.combat.contacts > 0 and inputs.combat.supporters == 0)
                    reward = -TICK_PENALTY + PROGRESS_REWARD * progress - DAMAGE_PENALTY * own_damage
                    reward -= carrier_stall_cost(before, inputs, chosen_action)
                    support_credit = support_progress_credit(before, char.pos, inputs, done_team or not char.is_alive)
                    reward += support_credit
                    entry_counts['support_plan_ticks'] += int(inputs.support is not None)
                    entry_counts['support_shaping_reward'] += support_credit
                    reward += KILL_REWARD * (char.round_kills - previous_kills[name]) + TEAM_DAMAGE_REWARD * min(200., enemy_damage)
                    reward += STATIONARY_HIT_REWARD * sum(s["hit"] and not s["shooter"].moved_this_tick
                        for s in getattr(game, "last_shots", ()) if s["shooter"] is char)
                    shots = [s for s in getattr(game, 'last_shots', ()) if s['shooter'].team == 'A']
                    reward += COOPERATIVE_HIT_REWARD * sum(s['hit'] and s['shooter'] is char
                        and len({id(x['shooter']) for x in shots if x['target'] is s['target']}) >= 2 for s in shots)
                    if decision_snapshot is not None:
                        from touyama_v3.tv3_attacker_entry_coordination import shared_fire_score
                        public_targets = tuple(s.position for s in decision_snapshot.sightings)
                        reward += SHARED_FIRE_PROGRESS_REWARD * (
                            .98 * shared_fire_score(scenario, next_snapshot, public_targets)
                            - shared_fire_score(scenario, decision_snapshot, public_targets))
                    reward -= group_separation_cost(scenario, next_snapshot, char.pos, before, inputs.combat,
                        decision_plan, decision_has_flanks, damage=own_damage, terminal=done_team)
                    reward -= DEATH_PENALTY * int(not char.is_alive)
                    reward -= ABILITY_USE_PENALTY * int(chosen_action.kind in ("ABILITY", "ULTIMATE"))
                    reward += PLANT_PROGRESS_REWARD * max(0, char.plant_timer - before.plant_progress)
                    for sighting in sightings:
                        contact = (before.slot, sighting.enemy_id)
                        if contact not in contacts and sighting.enemy_id in enemy_before and scenario.clear(
                                before.position, enemy_before[sighting.enemy_id]):
                            contacts.add(contact)
                            # Credit anticipatory facing once at a geometrically relevant first contact.
                            reward += first_contact_preaim(scenario, before.position,
                                enemy_before[sighting.enemy_id], chosen_action.facing)
                    done = done_team or not char.is_alive
                    next_inputs = controller.inputs.get(name)
                    next_obs = next_inputs.observation if next_inputs is not None and not done else np.zeros_like(inputs.observation)
                    next_mask = next_inputs.mask.copy() if next_inputs is not None and not done else np.ones(ACTION_DIM, bool)
                    if next_inputs is None and not done:
                        done = True
                    transitions.append([inputs.observation.astype(np.float16), action_index, reward,
                        next_obs.astype(np.float16), next_mask, float(done), inputs.teacher, inputs.mask.copy()])
                    if name in last_indices and transitions[last_indices[name]][5]>.5:
                        decision_counts[name]=0  # Count only the actor's connected episode.
                    last_indices[name] = len(transitions) - 1
                    decision_counts[name] += 1
                    if chosen_action.kind == "ABILITY":
                        utility_indices[(decision_tick, name, before.ability_name)] = len(transitions)-1
                    reward_sum += reward
                if audit:
                    for record in audit.utilities:
                        key = record["tick"], record["name"], record["ability"]
                        if record["resolved"] and not record.get("credited") and key in utility_indices:
                            credit = utility_effect_credit(record)
                            transitions[utility_indices[key]][2] += credit
                            reward_sum += credit
                            record["credited"] = True
                if done_team:
                    terminal = {"planted": bool(game.is_planted), "tick": int(game.battle_tick),
                        "preplant_round_won":bool(not game.is_planted and game.attacker_wins>score_before),
                        "alive": sum(c.is_alive for c in own), "enemy_alive": sum(c.is_alive for c in enemy),
                        "remaining_hp": float(sum(max(0., c.hp) for c in own if c.is_alive)),
                        "initial_team_hp": initial_team_hp,
                        "initial_alive": initial_alive, "initial_enemy_alive": initial_enemy_alive,
                        "initial_abilities": initial_abilities,
                        "remaining_abilities_alive": sum(charges[n] for n, c in current.items() if c.is_alive),
                        # Evaluation label only, read after the public action has executed.
                        "enemy_remaining_abilities_alive": sum(int(getattr(c,c.ability_name.lower()+'_charges',0))
                            for c in enemy if c.is_alive),
                        "damage": damage, "uses": uses,
                        "site": scenario.site_of(game.planted_pos) if game.is_planted else None,
                        "reason": "planted" if game.is_planted else "attacker_eliminated" if not any(c.is_alive for c in own)
                                  else "timeout" if game.round_timer <= 0 else "defender_eliminated"}
                    reserve = min(1., terminal["remaining_abilities_alive"] / initial_abilities) if initial_abilities else 0.
                    reward_sum += apply_terminal_bonus(transitions, last_indices,
                        {n: c.is_alive for n, c in current.items()}, terminal, initial_alive, reserve,decision_counts)
            if terminal is None or game.current_round != round_number or not game.round_over:
                raise RuntimeError("Plant block ended without a real terminal outcome")
            rounds.append({"round": round_number, "preset": preset_name, **terminal,
                "target_site": controller.route.site if controller.route else None,
                "route": controller.route.key if controller.route else None,
                "branches": controller.route.branches if controller.route else (),
                "replans": controller.replans, "route_decisions": controller.route_planner.events,
                "actual_paths": paths,
                "actual_branches": {name: branch_sequence(scenario, path) for name, path in paths.items()},
                "actual_site_entries": {name: [(path[i-1], p) for i, p in enumerate(path)
                    if i and scenario.grid[p] == 2 and scenario.grid[path[i-1]] != 2] for name, path in paths.items()},
                "combat_diagnostics": audit.report() if audit else None,
                "site_entry_diagnostics": dict(entry_counts),
                "reward": reward_sum})
            if round_callback is not None:
                round_callback(rounds[-1])
            history.append({"site": terminal["site"], "winner": "A" if game.attacker_wins > score_before else "D"})
            if round_number < start_round + 11:
                game.current_round += 1
                game.init_round()
    return rounds, transitions


def summarize_plant(rounds):
    metrics = summarize(rounds)
    metrics.update(low_damage_summary(rounds))
    metrics.update(plant_advantage_summary(rounds))
    metrics["mean_reward"] = float(np.mean([r["reward"] for r in rounds])) if rounds else 0.
    metrics["by_site"] = {}
    for side in ("L", "R"):
        subset = [r for r in rounds if r["target_site"] == side]
        metrics["by_site"][side] = {**summarize(subset), **low_damage_summary(subset), **plant_advantage_summary(subset)} if subset else {"rounds": 0, "plant_rate": None}
    return metrics


def low_damage_summary(rounds):
    count = sum(r['planted'] and r['alive'] >= LOW_DAMAGE_MIN_SURVIVORS
                and r.get('remaining_hp', 0.) / max(1., r.get('initial_team_hp', 1.)) >= LOW_DAMAGE_HP_FRACTION
                for r in rounds)
    return {'low_damage_plants': count, 'low_damage_plant_rate': count / len(rounds) if rounds else 0.}


def plant_advantage_summary(rounds):
    count=len(rounds)
    planted=[r for r in rounds if r['planted']]
    competitive=[r for r in planted if r['alive']>=r['enemy_alive']]
    advantaged=[r for r in planted if r['alive']>r['enemy_alive']]
    complete=all(r.get('enemy_remaining_abilities_alive') is not None and r.get('remaining_abilities_alive') is not None for r in rounds)
    ability_wins=sum(r['remaining_abilities_alive']>r['enemy_remaining_abilities_alive'] for r in competitive) if complete else None
    return dict(advantaged_plants=len(advantaged),advantaged_plant_rate=len(advantaged)/count if count else 0.,
        competitive_plants=len(competitive),competitive_plant_rate=len(competitive)/count if count else 0.,
        disadvantaged_plants=len(planted)-len(competitive),ability_labels_complete=complete,
        ability_advantage_plants=ability_wins,
        ability_advantage_plant_rate=ability_wins/count if complete and count else (0. if complete else None))


def best_rank(metrics):
    required=BEST_SELECTION_PRIORITIES
    if any(metrics.get(k) is None for k in required):
        raise ValueError('Best selection needs plant-time enemy counts/resources; re-evaluate the saved weights')
    return tuple(metrics[k] for k in required)+(-metrics['mean_damage'],-(metrics['mean_plant_ticks'] or 1000))


def exploration_values(completed_sets):
    fraction = min(1., completed_sets / EXPLORATION_DECAY_SETS)
    return (EPSILON_START + fraction * (EPSILON_END - EPSILON_START),
            TEACHER_START + fraction * (TEACHER_END - TEACHER_START))


def evaluate(opponent, scenario, analysis, policy, config, seeds, *, record_path=None, completed=None):
    rounds = []
    presets = eligible_attacker_presets(config["eval_presets"], opponent)
    plan = []
    for i, seed in enumerate(seeds):
        preset = presets[i % len(presets)]
        for start in config.get('evaluation_start_rounds', (1,)):
            records, _ = rollout(opponent, scenario, analysis, policy, config, seed, preset_name=preset, start_round=start)
            if record_path is not None:
                with Path(record_path).open('a',encoding='utf-8') as out:
                    for record in records:
                        out.write(json.dumps({'set':completed,'seed':seed,'preset':preset,
                            'teacher_probability':0,'epsilon':0,**record},ensure_ascii=False)+'\n')
            rounds.extend(records)
            plan.append({"seed": seed, "preset": preset, "start_round": start})
    return {"selected": summarize_plant(rounds), "seeds": list(seeds), "roster_plan": plan}


def log_evaluation(logger, opponent, evaluation, completed):
    metrics = evaluation["selected"]
    logger.info("[%s][実力評価 set=%d][教師・探索なし] %s", opponent, completed,
                format_summary(metrics), extra={"highlight_evaluation": True})
    for side, row in metrics["by_site"].items():
        if row["rounds"]:
            logger.info("[%s][サイト別評価 %s] %s", opponent, side, format_summary(row))
        else:
            logger.info("[%s][サイト別評価 %s] analysis未選択（評価対象なし）", opponent, side)
    logger.info("[%s][評価詳細] %s", opponent, json.dumps(evaluation, ensure_ascii=False))
    logger.info("[%s][少被害設置] %d/%d (%.1f%%、%d人生存・HP%.0f%%以上)", opponent,
                metrics['low_damage_plants'], metrics['rounds'], 100*metrics['low_damage_plant_rate'],
                LOW_DAMAGE_MIN_SURVIVORS, 100*LOW_DAMAGE_HP_FRACTION)
    logger.info('[%s][best選定] 人数優勢の設置=%d/%d 同人数以上の有効設置=%d/%d 有効設置時のアビリティ優勢=%s/%d',
        opponent,metrics['advantaged_plants'],metrics['rounds'],metrics['competitive_plants'],metrics['rounds'],
        metrics['ability_advantage_plants'],metrics['rounds'])


def save_latest(path, policy, target, optimizer, replay, completed, opponent, schema, config, analysis_hash, rng, last_evaluation):
    packed = [[torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for v in row] for row in replay]
    atomic_save(path, {"schema": schema, "opponent": opponent, "config": config, "analysis_hash": analysis_hash,
        "model": policy.state_dict(), "target": target.state_dict(), "optimizer": optimizer.state_dict(),
        "completed_sets": completed, "trained_rounds": completed * 12, "replay": packed,
        "rng": json.dumps(rng.bit_generator.state), "last_evaluation": last_evaluation,
        "selection_rule": BEST_SELECTION_POLICY,
        "warm_start_hash": getattr(policy,'warm_start_hash',None)})


def review_run_directory(directory,version):
    return Path(directory)/'review_checkpoints'/f'v{version}_{uuid.uuid4().hex}'


def save_review_weights(directory,policy,completed,opponent,schema,config,analysis_hash,started):
    path=Path(directory)/f'set_{completed:06d}.pt'
    atomic_save(path,dict(schema=schema,opponent=opponent,config=config,analysis_hash=analysis_hash,
        model=policy.state_dict(),completed_sets=completed,trained_rounds=completed*12,
        warm_start_hash=getattr(policy,'warm_start_hash',None),
        run_started_at_completed_set=started,evaluation_status='not_evaluated',
        checkpoint_kind='diagnostic_weights_only'))
    return path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-workers", type=int, default=MAX_PARALLEL_WORKERS)
    p.add_argument("--sets", type=int, default=TRAINING_SETS)
    p.add_argument("--total-sets", type=int, default=TOTAL_TRAINING_SET_LIMIT)
    p.add_argument("--opponents", nargs="+", choices=tuple(OPPONENTS), default=list(TARGET_OPPONENTS))
    p.add_argument("--train-presets", nargs="+", default=list(TRAINING_PRESETS))
    p.add_argument("--eval-presets", nargs="+", default=list(EVALUATION_PRESETS))
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME_TRAINING)
    p.add_argument("--eval-only", action=argparse.BooleanOptionalAction, default=EVALUATION_ONLY)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--eval-every", type=int, default=EVALUATION_INTERVAL)
    p.add_argument("--eval-seeds", type=int, default=EVALUATION_SEED_COUNT)
    p.add_argument("--updates", type=int, default=OPTIMIZER_UPDATES)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--replay-size", type=int, default=REPLAY_SIZE)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    p.add_argument("--max-round-steps", type=int, default=MAX_ROUND_STEPS)
    p.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIRECTORY)
    p.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    p.add_argument("--best-dir", type=Path, default=BEST_DIRECTORY)
    p.add_argument("--log-dir", type=Path, default=LOG_DIRECTORY)
    p.add_argument("--describe", action="store_true")
    return p


def remaining_training_sets(additional, completed, total_limit=None):
    if total_limit is not None and total_limit < 1:
        raise ValueError('TOTAL_TRAINING_SET_LIMIT must be positive or None')
    return additional if total_limit is None else max(0, min(additional, total_limit-completed))


def main(argv=None):
    args = parser().parse_args(argv)
    counts = ("sets", "eval_seeds", "updates", "batch_size", "replay_size", "torch_threads", "max_round_steps")
    if min(getattr(args, k) for k in counts) < 1 or args.eval_every < 0 or args.learning_rate <= 0:
        raise ValueError("Counts must be positive; evaluation interval >= 0")
    if args.eval_only and not args.resume:
        raise ValueError("Evaluation-only requires RESUME_TRAINING=True")
    if len(args.opponents) != len(set(args.opponents)):
        raise ValueError("Duplicate opponents")
    if (not 0 <= EPSILON_END <= EPSILON_START <= 1 or not 0 <= TEACHER_END <= TEACHER_START <= 1
            or not 0 <= ROUTE_EXPLORATION_END <= ROUTE_EXPLORATION_START <= 1
            or min(ROUTE_EXPLORATION_PROBE_STEPS, ROUTE_EXPLORATION_PROBE_TICKS) < 1
            or EXPLORATION_DECAY_SETS < 1):
        raise ValueError("Invalid exploration/demonstration schedule")
    torch.set_num_threads(args.torch_threads)
    scenario = Scenario()
    schema = policy_schema(scenario)
    for opponent in args.opponents:
        eligible_attacker_presets(args.train_presets, opponent)
        eligible_attacker_presets(args.eval_presets, opponent)
    config = {"seed": args.seed, "train_presets": args.train_presets, "eval_presets": args.eval_presets,
        "roster_scope": "touyama_fixed_five_independent_seeds", "eval_seeds": args.eval_seeds,
        "max_round_steps": args.max_round_steps, "rewards": reward_config(),
        "schedule": [EPSILON_START, EPSILON_END, TEACHER_START, TEACHER_END, EXPLORATION_DECAY_SETS],
        "route_exploration": dict(scope='all_routes', round_start_only=True,
            start=ROUTE_EXPLORATION_START, end=ROUTE_EXPLORATION_END,
            decay_sets=EXPLORATION_DECAY_SETS, probe_steps=ROUTE_EXPLORATION_PROBE_STEPS,
            probe_ticks=ROUTE_EXPLORATION_PROBE_TICKS),
        "demonstration_weight": DEMONSTRATION_WEIGHT, "formation": [GROUP_DISTANCE_LIMIT, GROUP_SEPARATION_PENALTY],
        "learning_by_opponent": {o: learning_settings(o, args.updates) for o in OPPONENTS},
        "formation_context": "executed_action_tick_v1",
        "low_damage_conditions": [LOW_DAMAGE_MIN_SURVIVORS, LOW_DAMAGE_HP_FRACTION]}
    config['training_start_rounds'] = list(TRAINING_START_ROUNDS)
    config['evaluation_start_rounds'] = list(EVALUATION_START_ROUNDS)
    config['failure_discount_compensation'] = FAILURE_DISCOUNT_COMPENSATION
    config['terminal_success_condition'] = 'all_plants_positive_man_ratio_or_enemy_elimination_v1'
    config['warm_start_from_best'] = WARM_START_FROM_BEST
    frozen = {opponent: load_frozen_analysis(args.analysis_dir.resolve(), opponent, scenario) for opponent in args.opponents}
    if args.describe:
        print(json.dumps({"schema": schema, "config": config,
                          "analysis_hashes": {k: v[1] for k, v in frozen.items()}}, ensure_ascii=False, indent=2))
        return 0
    data_dir, log_dir, best_dir = args.data_dir.resolve(), args.log_dir.resolve(), args.best_dir.resolve()
    from touyama_v3.tv3_training_parallel import run_opponent_workers
    status = run_opponent_workers(__file__, argv, args.opponents, args.max_workers, log_dir=log_dir)
    if status is not None:
        return status
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tv3.attacker_plant")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console, file_handler = logging.StreamHandler(), logging.FileHandler(log_dir / "training.log", mode="w", encoding="utf-8")
    formatter = ConsoleFormatter("%(asctime)s %(message)s", color_enabled=COLOR_CONSOLE and console_supports_color(console.stream))
    # Use this script's color constant while sharing Windows-console support.
    formatter.evaluation_color = EVALUATION_CONSOLE_COLOR
    console.setFormatter(formatter)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    handlers = (console, file_handler)
    for handler in handlers:
        logger.addHandler(handler)
    logger.info("attacker plant: analysis固定、相手別共有モデル、左右の強制均等化なし、guard対象外")
    logger.info("汎用AI: セットごとに学習編成を変更し、別編成で評価。学習=%s 評価=%s", args.train_presets, args.eval_presets)
    logger.info("[実力評価]（緑色）が教師・探索なしのプラント性能。[学習]は行動探索と教師を含みます。")
    logger.info("設定=%s", json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, ensure_ascii=False))
    try:
        for opponent in args.opponents:
            index = list(OPPONENTS).index(opponent)
            analysis, analysis_hash, _ = frozen[opponent]
            seed_all(args.seed + index)
            policy = PlantDQN()
            target = copy.deepcopy(policy)
            target.eval()
            optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
            replay = deque(maxlen=args.replay_size)
            rng = np.random.default_rng(args.seed + index)
            completed = 0
            reference = None
            directory = data_dir / opponent
            latest = directory / "latest.pt"
            best_path = best_dir / opponent / "attacker_plant_best.pt"
            if args.resume:
                state = load_checkpoint(latest)
                if "attacker_preset" in state.get("config", {}):
                    raise ValueError("This checkpoint was trained with one fixed roster. Generic multi-roster training requires a fresh run; existing files have not been changed.")
                if state["schema"] != schema or state["opponent"] != opponent or state["config"] != config or state["analysis_hash"] != analysis_hash:
                    raise ValueError(f"Resume map/config/analysis mismatch: {latest}")
                policy.load_state_dict(state["model"])
                target.load_state_dict(state["target"])
                optimizer.load_state_dict(state["optimizer"])
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
                replay.extend([[v.numpy() if isinstance(v, torch.Tensor) else v for v in row] for row in state["replay"]])
                rng.bit_generator.state = json.loads(state["rng"])
                completed, reference = state["completed_sets"], state.get("last_evaluation")
                policy.warm_start_hash=state.get('warm_start_hash')
            elif WARM_START_FROM_BEST and best_path.exists():
                initial,initial_hash=warm_start_source(best_path,schema,analysis_hash,opponent,data_dir/'warm_start_sources')
                if initial is not None:
                    policy.load_state_dict(initial['model'])
                    target.load_state_dict(initial['model'])
                    policy.warm_start_hash=initial_hash
                    logger.info('[%s] best v%d set=%dの重みからv%dの入力・教師・報酬で学習。optimizer/replayは新規、通算setは0から。',
                        opponent,initial['schema']['version'],initial['completed_sets'],schema['version'])
                else:
                    logger.info('[%s] 保存bestの観測/analysis依存が異なるため、重み初期化は新規。',opponent)
            policy.eval()
            seeds = [args.seed + 800_000_000 + index * 1_000_000 + i * 100 for i in range(args.eval_seeds)]
            old_evaluation = None
            if (args.resume or getattr(policy,'warm_start_hash',None)) and best_path.exists():
                from touyama_v3.tv3_guard_runtime import read_checkpoint
                old,old_hash = read_checkpoint(best_path)
                if (not warm_start_schema_compatible(old['schema'],schema)
                        or old['analysis_hash']!=analysis_hash or old['opponent']!=opponent):
                    raise ValueError(f"Best evaluation conditions differ: {best_path}")
                old_evaluation = old["evaluation"]
                if not args.eval_only and (old_evaluation['selected'].get('ability_advantage_plant_rate') is None
                        or old['config']!=config or old_evaluation['seeds']!=seeds
                        or old.get('selection_rule')!=BEST_SELECTION_POLICY):
                    # Historical summaries cannot reconstruct enemy charges. Re-evaluate
                    # the actual saved weights; do not invent labels or reset training.
                    cache_path=log_dir/f'{opponent}_best_reassessment.json'
                    cache_key=dict(model_hash=old_hash,config=config,seeds=seeds,selection=BEST_SELECTION_POLICY)
                    cached=json.loads(cache_path.read_text(encoding='utf8')) if cache_path.exists() else {}
                    if cached.get('key')==cache_key:
                        old_evaluation=cached['evaluation']
                    else:
                        with torch.random.fork_rng(devices=[]):
                            baseline=PlantDQN(old['schema']['version'])
                            baseline.load_state_dict(old['model'])
                        baseline.eval()
                        old_evaluation=evaluate(opponent,scenario,analysis,baseline,config,seeds,
                            record_path=log_dir/f'{opponent}_best_reassessment_rounds.jsonl',completed=old['completed_sets'])
                        cache_path.write_text(json.dumps(dict(key=cache_key,evaluation=old_evaluation),ensure_ascii=False),encoding='utf8')
                    logger.info('[%s] 新しい選定基準で保存best set=%dを再評価（学習更新なし）',opponent,old['completed_sets'])
                logger.info("[%s][保存best] 採用set=%d %s", opponent, old["completed_sets"], format_summary(old_evaluation["selected"]))
            if args.eval_only:
                result = evaluate(opponent, scenario, analysis, policy, config, seeds,
                    record_path=log_dir/f'{opponent}_evaluation_rounds.jsonl',completed=completed)
                log_evaluation(logger, opponent, result, completed)
                continue
            additional_sets = remaining_training_sets(args.sets, completed, args.total_sets)
            logger.info("[%s] 完了=%d 今回追加=%dセット 通算上限=%s analysis=%s",
                        opponent, completed, additional_sets, args.total_sets, analysis_hash)
            review_directory=None
            review_started=completed
            if SAVE_REVIEW_WEIGHT_CHECKPOINTS and additional_sets:
                review_directory=review_run_directory(directory,policy.execution_version)
                save_review_weights(review_directory,policy,completed,opponent,schema,config,analysis_hash,review_started)
                logger.info('[%s] 診断用重みを学習前・各セットで保存（未評価・optimizer/replayなし）: %s',opponent,review_directory)
            for additional in range(1, additional_sets + 1):
                epsilon, teacher = exploration_values(completed)
                fraction = min(1., completed / EXPLORATION_DECAY_SETS)
                route_exploration = ROUTE_EXPLORATION_START + fraction * (ROUTE_EXPLORATION_END - ROUTE_EXPLORATION_START)
                training_presets = eligible_attacker_presets(args.train_presets, opponent)
                preset = training_presets[completed % len(training_presets)]
                records, transitions = rollout(opponent, scenario, analysis, policy, config,
                    args.seed + index * 1_000_000 + (completed + 1) * 100, training=True,
                    epsilon=epsilon, teacher_probability=teacher, preset_name=preset, route_exploration=route_exploration,
                    start_round=TRAINING_START_ROUNDS[completed % len(TRAINING_START_ROUNDS)])
                replay.extend(transitions)
                settings = learning_settings(opponent, args.updates)
                loss = learn_plant(policy, target, optimizer, replay, rng, settings['updates'], args.batch_size,
                    settings['demonstration_weight'], settings['demonstration_kind_weight'], settings['carrier_sample_fraction'],
                    demonstration_teacher_variant=settings['demonstration_teacher_variant'])
                completed += 1
                save_latest(latest, policy, target, optimizer, replay, completed, opponent, schema, config, analysis_hash, rng, reference)
                if review_directory is not None:
                    save_review_weights(review_directory,policy,completed,opponent,schema,config,analysis_hash,review_started)
                metrics = summarize_plant(records)
                logger.info('[%s][学習 set=%d] 経路探索=%.1f%%（開始時の全候補）', opponent, completed, 100*route_exploration)
                logger.info("[%s][学習 set=%d][編成=%s][教師選択=%.1f%% 教師以外の探索=%.1f%%] 直近実力=%s | %s loss=%s", opponent,
                    completed, preset, 100 * teacher, 100 * epsilon, format_last_evaluation(reference),
                    format_summary(metrics, "収集プラント成功率"), loss)
                with (log_dir / f"{opponent}_rounds.jsonl").open("w" if additional == 1 else "a", encoding="utf-8") as out:
                    for record in records:
                        out.write(json.dumps({"set": completed, **record}, ensure_ascii=False) + "\n")
                if (args.eval_every and completed % args.eval_every == 0) or additional == additional_sets:
                    result = evaluate(opponent, scenario, analysis, policy, config, seeds,
                        record_path=log_dir/f'{opponent}_evaluation_rounds.jsonl',completed=completed)
                    log_evaluation(logger, opponent, result, completed)
                    reference = {"set": completed, "evaluation": result}
                    save_latest(latest, policy, target, optimizer, replay, completed, opponent, schema, config, analysis_hash, rng, reference)
                    evaluated={"schema": schema, "opponent": opponent, "config": config,
                            "analysis_hash": analysis_hash, "model": policy.state_dict(), "completed_sets": completed,
                            "trained_rounds": completed * 12, "evaluation": result,
                            "selection_rule": BEST_SELECTION_POLICY,
                            "warm_start_hash": getattr(policy,'warm_start_hash',None)}
                    if SAVE_EVALUATION_CHECKPOINTS:
                        atomic_save(directory/'evaluation_checkpoints'/f'set_{completed:06d}.pt',evaluated)
                    if old_evaluation is None or best_rank(result["selected"]) > best_rank(old_evaluation["selected"]):
                        atomic_save(best_path,evaluated)
                        old_evaluation = result
                        logger.info("[%s] best更新 採用set=%d 保存=%s", opponent, completed, best_path)
                    else:
                        logger.info("[%s] best維持=%s", opponent, best_path)
    except KeyboardInterrupt:
        logger.info("中断。完了セットはlatest保存済み。RESUME_TRAINING=Trueで再開できます。")
        return 130
    except Exception:
        logger.exception("プラント学習失敗。完了セットのlatestは保持しています。")
        raise
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
