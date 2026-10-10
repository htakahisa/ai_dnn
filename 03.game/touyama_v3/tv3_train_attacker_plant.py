"""Train opponent-specific attacker plant policies with frozen analysis routes."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 通常設定：このファイル冒頭を編集し、touyama_v3からオプションなしで実行。
TRAINING_SETS = 100  # 各相手への追加セット。1セット=12ラウンド。
TOTAL_TRAINING_SET_LIMIT = 100 # 最大のセット。50セットから200追加としても、100セット目でとまる
MAX_PARALLEL_WORKERS = 6  # 相手AIごとの最大同時実行数。1で順次実行。
RESUME_TRAINING = False

TARGET_OPPONENTS = ("gc_v1", "concon_v1", "omoko_v1", "fnatic_v3", "frc_v1", "toru_ai_v4")
#TARGET_OPPONENTS = ("frc_v1",)  # 1種類でも末尾のカンマが必要。
TRAINING_PRESETS = ("Touyama Gaming",)
EVALUATION_PRESETS = ("Touyama Gaming",)  # 同じ5人・独立seedで評価。
EVALUATION_ONLY = False
RANDOM_SEED = 42
EVALUATION_INTERVAL = 10
EVALUATION_SEED_COUNT = 3
OPTIMIZER_UPDATES = 200
BATCH_SIZE = 64
REPLAY_SIZE = 60000
LEARNING_RATE = .001
EPSILON_START = .20
EPSILON_END = .03
TEACHER_START = .95
TEACHER_END = .10
EXPLORATION_DECAY_SETS = 60
DEMONSTRATION_WEIGHT = .05
DEMONSTRATION_KIND_WEIGHT = .25
CARRIER_SAMPLE_FRACTION = .50
FRC_DEMONSTRATION_WEIGHT = .15
FRC_DEMONSTRATION_KIND_WEIGHT = 1.
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
PLANT_FAILURE_REWARD = -6.
SURVIVOR_REWARD = .30
RESOURCE_REWARD = .30
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
LOW_DAMAGE_PLANT_REWARD = 3.
PLANT_HP_RETENTION_REWARD = 1.
COOPERATIVE_HIT_REWARD = .08
SHARED_FIRE_PROGRESS_REWARD = .04
CARRIER_STALL_PENALTY = .08

import argparse
from collections import deque, Counter
import contextlib
import copy
import json
import logging
import os

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
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
from touyama_v3.tv3_attacker_site_entry import mission_progress
from touyama_v3.tv3_attacker_rosters import eligible_attacker_presets
from touyama_v3.tv3_checkpoint import load_checkpoint


def reward_config():
    return {key: value for key, value in globals().items() if key.endswith("_REWARD") or key.endswith("_PENALTY")}


def carrier_stall_cost(before, inputs, action):
    """Penalize avoidable idle ticks, preserving combat and coordinated waits."""
    if (not before.has_spike or action.kind != 'STAY' or before.blind
            or before.plant_progress or inputs.combat is None
            or inputs.combat.contacts or inputs.combat.reason != 'advance'
            or inputs.actions[inputs.teacher].kind == 'STAY'):
        return 0.
    return CARRIER_STALL_PENALTY if inputs.observation[-2:].max() > .5 else 0.


def learning_settings(opponent, updates):
    frc = opponent == 'frc_v1'
    return {'demonstration_weight': FRC_DEMONSTRATION_WEIGHT if frc else DEMONSTRATION_WEIGHT,
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


def terminal_bonus(terminal, initial_alive, reserve):
    hp_fraction = terminal.get('remaining_hp', 0.) / max(1., terminal.get('initial_team_hp', 1.))
    low_damage = (terminal['alive'] >= LOW_DAMAGE_MIN_SURVIVORS and hp_fraction >= LOW_DAMAGE_HP_FRACTION)
    return (PLANT_SUCCESS_REWARD * terminal["alive"] / max(1, initial_alive)
            + SURVIVOR_REWARD * terminal["alive"] + RESOURCE_REWARD * reserve
            + PLANT_HP_RETENTION_REWARD * min(1., hp_fraction) + LOW_DAMAGE_PLANT_REWARD * low_damage
            if terminal["planted"] else PLANT_FAILURE_REWARD)


def apply_terminal_bonus(transitions, indices, alive, terminal, initial_alive, reserve):
    bonus = terminal_bonus(terminal, initial_alive, reserve)
    total = 0.
    for n, index in indices.items():
        if terminal["planted"] and not alive[n]:
            continue
        transitions[index][2] += bonus
        transitions[index][5] = 1.
        total += bonus
    return total


def rollout(opponent, scenario, analysis, policy, config, seed, *, preset_name,
            training=False, epsilon=0., teacher_probability=0., tick_callback=None):
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
        team = DualRoleTeamAI("Touyama v3 plant trainer", lambda: controller, DefaultDefenderController)
        game = VisualFPSBattle(scenario.maze, team, _build_team_ai(OPPONENTS[opponent][0], device="cpu"), headless=True,
            attacker_roster=list(own_preset.players), defender_roster=list(enemy_preset.players),
            spike_holder_name=own_preset.spike_holder, defender_spike_holder_name=enemy_preset.spike_holder,
            attacker_igl_name=own_preset.igl, defender_igl_name=enemy_preset.igl,
            attacker_team_name=own_preset.name, defender_team_name=enemy_preset.name, disable_side_swap=True)
        game.stop_after_round, game.analytics_tracker = True, None
        game._record_replay_frame = lambda: None
        relocate_debug_logs(game.defender_controller, None)
        for round_number in range(1, 13):
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
            utility_indices = {}
            contacts = set()
            entry_counts = Counter()
            entry_launched = False
            score_before = game.attacker_wins
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
                    next_obs = next_inputs.observation if next_inputs is not None and not done else np.zeros(OBS_DIM, np.float32)
                    next_mask = next_inputs.mask.copy() if next_inputs is not None and not done else np.ones(ACTION_DIM, bool)
                    if next_inputs is None and not done:
                        done = True
                    transitions.append([inputs.observation.astype(np.float16), action_index, reward,
                        next_obs.astype(np.float16), next_mask, float(done), inputs.teacher, inputs.mask.copy()])
                    last_indices[name] = len(transitions) - 1
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
                        "alive": sum(c.is_alive for c in own), "enemy_alive": sum(c.is_alive for c in enemy),
                        "remaining_hp": float(sum(max(0., c.hp) for c in own if c.is_alive)),
                        "initial_team_hp": initial_team_hp,
                        "initial_alive": initial_alive, "initial_enemy_alive": initial_enemy_alive,
                        "initial_abilities": initial_abilities,
                        "remaining_abilities_alive": sum(charges[n] for n, c in current.items() if c.is_alive),
                        "damage": damage, "uses": uses,
                        "site": scenario.site_of(game.planted_pos) if game.is_planted else None,
                        "reason": "planted" if game.is_planted else "attacker_eliminated" if not any(c.is_alive for c in own)
                                  else "timeout" if game.round_timer <= 0 else "defender_eliminated"}
                    reserve = min(1., terminal["remaining_abilities_alive"] / initial_abilities) if initial_abilities else 0.
                    reward_sum += apply_terminal_bonus(transitions, last_indices,
                        {n: c.is_alive for n, c in current.items()}, terminal, initial_alive, reserve)
            if terminal is None or game.current_round != round_number or not game.round_over:
                raise RuntimeError("Plant block ended without a real terminal outcome")
            rounds.append({"round": round_number, "preset": preset_name, **terminal,
                "target_site": controller.route.site if controller.route else None,
                "route": controller.route.key if controller.route else None,
                "branches": controller.route.branches if controller.route else (),
                "replans": controller.replans, "route_decisions": controller.route_planner.events,
                "combat_diagnostics": audit.report() if audit else None,
                "site_entry_diagnostics": dict(entry_counts),
                "reward": reward_sum})
            history.append({"site": terminal["site"], "winner": "A" if game.attacker_wins > score_before else "D"})
            if round_number < 12:
                game.current_round += 1
                game.init_round()
    return rounds, transitions


def summarize_plant(rounds):
    metrics = summarize(rounds)
    metrics.update(low_damage_summary(rounds))
    metrics["mean_reward"] = float(np.mean([r["reward"] for r in rounds])) if rounds else 0.
    metrics["by_site"] = {}
    for side in ("L", "R"):
        subset = [r for r in rounds if r["target_site"] == side]
        metrics["by_site"][side] = {**summarize(subset), **low_damage_summary(subset)} if subset else {"rounds": 0, "plant_rate": None}
    return metrics


def low_damage_summary(rounds):
    count = sum(r['planted'] and r['alive'] >= LOW_DAMAGE_MIN_SURVIVORS
                and r.get('remaining_hp', 0.) / max(1., r.get('initial_team_hp', 1.)) >= LOW_DAMAGE_HP_FRACTION
                for r in rounds)
    return {'low_damage_plants': count, 'low_damage_plant_rate': count / len(rounds) if rounds else 0.}


def best_rank(metrics):
    return (metrics.get('low_damage_plant_rate', metrics.get('viable_plant_rate', metrics['plant_rate'])),
            metrics.get("viable_plant_rate", metrics["plant_rate"]), metrics["plant_rate"], metrics["mean_alive"], -metrics["mean_damage"],
            metrics["ability_reserve_rate"] if metrics["ability_reserve_rate"] is not None else 0.,
            -metrics["mean_ability_uses"], -(metrics["mean_plant_ticks"] or 1000))


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
        records, _ = rollout(opponent, scenario, analysis, policy, config, seed, preset_name=preset)
        if record_path is not None:
            with Path(record_path).open('a',encoding='utf-8') as out:
                for record in records:
                    out.write(json.dumps({'set':completed,'seed':seed,'preset':preset,
                        'teacher_probability':0,'epsilon':0,**record},ensure_ascii=False)+'\n')
        rounds.extend(records)
        plan.append({"seed": seed, "preset": preset})
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


def save_latest(path, policy, target, optimizer, replay, completed, opponent, schema, config, analysis_hash, rng, last_evaluation):
    packed = [[torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for v in row] for row in replay]
    atomic_save(path, {"schema": schema, "opponent": opponent, "config": config, "analysis_hash": analysis_hash,
        "model": policy.state_dict(), "target": target.state_dict(), "optimizer": optimizer.state_dict(),
        "completed_sets": completed, "trained_rounds": completed * 12, "replay": packed,
        "rng": json.dumps(rng.bit_generator.state), "last_evaluation": last_evaluation})


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
    if not 0 <= EPSILON_END <= EPSILON_START <= 1 or not 0 <= TEACHER_END <= TEACHER_START <= 1 or EXPLORATION_DECAY_SETS < 1:
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
        "demonstration_weight": DEMONSTRATION_WEIGHT, "formation": [GROUP_DISTANCE_LIMIT, GROUP_SEPARATION_PENALTY],
        "learning_by_opponent": {o: learning_settings(o, args.updates) for o in OPPONENTS},
        "formation_context": "executed_action_tick_v1",
        "low_damage_conditions": [LOW_DAMAGE_MIN_SURVIVORS, LOW_DAMAGE_HP_FRACTION]}
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
            policy.eval()
            seeds = [args.seed + 800_000_000 + index * 1_000_000 + i * 100 for i in range(args.eval_seeds)]
            old_evaluation = None
            if args.resume and best_path.exists():
                old = load_checkpoint(best_path)
                if any(old[k] != v for k, v in (("schema", schema), ("config", config), ("analysis_hash", analysis_hash))):
                    raise ValueError(f"Best evaluation conditions differ: {best_path}")
                if old["evaluation"]["seeds"] != seeds:
                    raise ValueError("Best evaluation seeds differ")
                old_evaluation = old["evaluation"]
                logger.info("[%s][保存best] 採用set=%d %s", opponent, old["completed_sets"], format_summary(old_evaluation["selected"]))
            if args.eval_only:
                result = evaluate(opponent, scenario, analysis, policy, config, seeds,
                    record_path=log_dir/f'{opponent}_evaluation_rounds.jsonl',completed=completed)
                log_evaluation(logger, opponent, result, completed)
                continue
            additional_sets = remaining_training_sets(args.sets, completed, args.total_sets)
            logger.info("[%s] 完了=%d 今回追加=%dセット 通算上限=%s analysis=%s",
                        opponent, completed, additional_sets, args.total_sets, analysis_hash)
            for additional in range(1, additional_sets + 1):
                epsilon, teacher = exploration_values(completed)
                training_presets = eligible_attacker_presets(args.train_presets, opponent)
                preset = training_presets[completed % len(training_presets)]
                records, transitions = rollout(opponent, scenario, analysis, policy, config,
                    args.seed + index * 1_000_000 + (completed + 1) * 100, training=True,
                    epsilon=epsilon, teacher_probability=teacher, preset_name=preset)
                replay.extend(transitions)
                settings = learning_settings(opponent, args.updates)
                loss = learn_plant(policy, target, optimizer, replay, rng, settings['updates'], args.batch_size,
                    settings['demonstration_weight'], settings['demonstration_kind_weight'], settings['carrier_sample_fraction'])
                completed += 1
                save_latest(latest, policy, target, optimizer, replay, completed, opponent, schema, config, analysis_hash, rng, reference)
                metrics = summarize_plant(records)
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
                    if old_evaluation is None or best_rank(result["selected"]) > best_rank(old_evaluation["selected"]):
                        atomic_save(best_path, {"schema": schema, "opponent": opponent, "config": config,
                            "analysis_hash": analysis_hash, "model": policy.state_dict(), "completed_sets": completed,
                            "trained_rounds": completed * 12, "evaluation": result,
                            "selection_rule": "low_damage_then_viable_plant_survival_damage_reserve_v3"})
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
