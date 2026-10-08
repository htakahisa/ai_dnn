"""Train new roster-independent Toru v4 search/retake policies in the real engine."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import deque
import contextlib
import copy
import json
import logging
import os
import time
import numpy as np
import torch

from toruAI_v4.tv4_scenario import Scenario, OPPONENTS
from toruAI_v4.tv4_train_analysis import legacy_root, seed_all, relocate_debug_logs
from toruAI_v4.tv4_defender_policy import DefenderDQN, OBS_DIM, ACTION_DIM, learn_dqn
from toruAI_v4.tv4_defender_controller import (
    ToruV4DefenderController, load_analyses, load_policy, policy_metadata, normalize_policy_schema, DEFENDER_BEST,
)

# 各相手モデルの1セット = その相手との12ラウンド。味方編成はセットごとに変える。
TRAINING_SETS = 100
EVALUATION_INTERVAL = 10
EVALUATION_SEED_COUNT = 3
PLANT_ADVANTAGE_REWARD = .10  # 味方生存人数 - 相手生存人数
RETAKE_COMBAT_RESERVE_TICKS = 20
RETAKE_SAFETY_MARGIN_TICKS = 5
SEARCH_EVALUATION_SCOPE = f"preplant_rally_abc_v1_combat{RETAKE_COMBAT_RESERVE_TICKS}_margin{RETAKE_SAFETY_MARGIN_TICKS}"
TRAINING_PRESETS = ("Gorigons", "EG2023", "Vision Strikers", "Furina Classic", "Fnatic2023",
                    "Touyama Gaming", "Omoko Gaming", "Ghost Champions", "Team Elites")
# 学習では使わない編成で評価。各seedで違う編成を選ぶ。
EVALUATION_PRESETS = ("Eine Kleine", "SUPES", "BBL")
REPLAY_SIZE = 60000
BATCH_SIZE = 64
UPDATES_PER_BLOCK = 100


def eligible_presets(names, opponent):
    from party_presets import get_preset
    enemy = get_preset(OPPONENTS[opponent][1])
    valid = []
    for name in names:
        own = get_preset(name)
        if own is None or len(own.players) != 5 or len(set(own.players)) != 5:
            raise ValueError(f"Invalid five-player preset: {name}")
        if not set(own.players) & set(enemy.players):
            valid.append(name)
    if not valid:
        raise ValueError(f"No disjoint defender preset for {opponent}")
    return valid


def near_plant_distances(scenario, plant):
    from grid_paths import distance_map
    goals = [(plant[0] + dr, plant[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if 0 <= plant[0] + dr < scenario.grid.shape[0] and 0 <= plant[1] + dc < scenario.grid.shape[1]
             and scenario.grid[plant[0] + dr, plant[1] + dc] != 1]
    maps = [distance_map(scenario.grid, p) for p in goals]
    distances = np.minimum.reduce([np.where(m >= 0, m, np.inf) for m in maps])
    return distances


def retake_arrival_readiness(distances, remaining_ticks):
    """Walking estimates to a/b/c rally areas; reserve entry/combat/defuse time."""
    from game_core import DEFUSE_REQUIRED_TICKS
    travel = [float(d) if np.isfinite(d) and d >= 0 else float("inf") for d in distances]
    budget = float(remaining_ticks) - DEFUSE_REQUIRED_TICKS - RETAKE_COMBAT_RESERVE_TICKS - RETAKE_SAFETY_MARGIN_TICKS
    deficits = [max(0., d - budget) for d in travel]
    late = sum(d > 0 for d in deficits)
    # One late member is enough for a substantial shared penalty. Relative
    # distance alone incurs no penalty when every member meets the budget.
    penalty = min(5., 2. * late + .1 * max(deficits, default=0.)) if late else 0.
    return dict(plant_ready_count=len(travel) - late, plant_late_count=late,
                plant_all_ready=bool(travel) and late == 0,
                plant_max_distance=max(travel) if travel else None,
                plant_min_margin=float(remaining_ticks) - DEFUSE_REQUIRED_TICKS - RETAKE_COMBAT_RESERVE_TICKS - max(travel) if travel else None,
                plant_travel_budget=budget, plant_arrival_penalty=penalty)


def retained_resources(chars, initial):
    total = sum(initial.values())
    if not total:
        return None
    remaining = sum(int(getattr(c, c.ability_name.lower() + "_charges", 0)) for c in chars if c.is_alive)
    # Regeneration can add charges; retention is bounded to the initial budget.
    return min(1., remaining / total)


def rollout(opponent, preset_name, scenario, search, retake, analyses, phase, seed, *,
            training=False, epsilon=0., teacher_probability=0., logger=None, round_callback=None, log_context=""):
    """Twelve real consecutive rounds. No teleporting to fabricated plant states."""
    from party_presets import get_preset
    from controllers import DefaultAttackerController
    from team_ai import DualRoleTeamAI
    from simulation_runtime import cpu_inference
    transitions, rounds = [], []
    with legacy_root(), open(os.devnull, "w", encoding="utf-8") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet), cpu_inference(enabled=True):
        from run_game import VisualFPSBattle, _build_team_ai
        seed_all(seed)
        own, enemy = get_preset(preset_name), get_preset(OPPONENTS[opponent][1])
        if set(own.players) & set(enemy.players):
            raise ValueError("Enemy and ally identities must be disjoint in the engine")
        controller = ToruV4DefenderController(opponent, scenario=scenario, search=search, retake=retake,
                                            analyses=analyses, training=True, seed=seed + 19)
        controller.training, controller.learning_phase = training, phase
        controller.stop_at_plant = phase == "search"
        controller.epsilon, controller.teacher_probability = epsilon, teacher_probability
        ai = DualRoleTeamAI("Toru AI v4", DefaultAttackerController, lambda: controller)
        game = VisualFPSBattle(scenario.maze, _build_team_ai(OPPONENTS[opponent][0]), ai, headless=True,
            attacker_roster=list(enemy.players), defender_roster=list(own.players),
            spike_holder_name=enemy.spike_holder, defender_spike_holder_name=own.spike_holder,
            attacker_igl_name=enemy.igl, defender_igl_name=own.igl, disable_side_swap=True,
            attacker_team_name=enemy.name, defender_team_name=own.name)
        game.stop_after_round, game.analytics_tracker = True, None
        game._record_replay_frame = lambda: None
        relocate_debug_logs(game.attacker_controller, None)
        for round_no in range(1, 13):
            last_indices = {}
            planted = search_closed = False
            plant_d = plant_a = plant_distance = None
            plant_reserve = None
            arrival = {}
            initial_charges = {str(c.name): int(getattr(c, c.ability_name.lower() + "_charges", 0))
                               for c in game.chars if c.team == "D"}
            skill_uses = 0
            reward_sum, steps, retake_steps = 0., 0, 0
            score_before = game.defender_wins
            while not game.round_over and not game.match_over:
                controller.decisions = {}
                hp = {str(c.name): c.hp for c in game.chars if c.team == "D"}
                kills = {str(c.name): c.round_kills for c in game.chars if c.team == "D"}
                previously_seen = set(controller.history.tracks)
                enemy_hp_before = sum(c.hp for c in game.chars if c.team == "A" and c.is_alive)
                game.step_tick()
                steps += 1
                if steps > 400:
                    raise RuntimeError(f"Round watchdog exceeded: {opponent} R{round_no}")
                retake_steps += int(planted)
                current = {str(c.name): c for c in game.chars if c.team == "D"}
                enemy_damage = max(0., enemy_hp_before - sum(c.hp for c in game.chars if c.team == "A" and c.is_alive))
                terminal = game.round_over or game.match_over or (phase == "search" and game.is_planted)
                if not terminal:
                    controller.prepare_team_tick()
                for name, (action_phase, action, inputs, before) in list(controller.decisions.items()):
                    if action_phase != phase or search_closed:
                        continue
                    char = current[name]
                    progress = float(np.clip(inputs.distances[before.position] - inputs.distances[tuple(char.pos)], -1, 1))
                    damage = max(0., hp[name] - char.hp)
                    reward = -.005 + .04 * progress - .01 * damage + .6 * (char.round_kills - kills[name])
                    reward += .001 * min(200., enemy_damage)  # Cooperative outcome reward, not an observation.
                    if inputs.actions[action].kind in ("ABILITY", "ULTIMATE"):
                        reward -= .02  # Avoid rewarding resource consumption by itself.
                        remaining = int(getattr(char, char.ability_name.lower() + "_charges", 0))
                        if inputs.actions[action].kind == "ABILITY" and remaining < before.charges:
                            skill_uses += 1
                            if phase == "search" and remaining == 0:
                                reward -= .08  # Last charge can be used, but has a retake opportunity cost.
                    reward -= 1.5 * int(not char.is_alive)
                    if phase == "retake":
                        reward += .12 * max(0, char.defuse_timer - before.defuse_progress)
                    else:
                        reward += .01 * sum(s.enemy_id not in previously_seen
                                          for s in controller.snapshot.sightings)
                    next_inputs = controller.inputs.get(name)
                    done = terminal or not char.is_alive or (phase == "search" and game.is_planted)
                    next_obs = next_inputs.observation if next_inputs is not None and not done else np.zeros(OBS_DIM, np.float32)
                    next_mask = next_inputs.mask.copy() if next_inputs is not None and not done else np.ones(ACTION_DIM, bool)
                    transitions.append([inputs.observation.astype(np.float16), action, reward,
                                        next_obs.astype(np.float16), next_mask, float(done),
                                        scenario.site_of(game.planted_pos) if action_phase == "retake" else None])
                    last_indices[name] = len(transitions) - 1
                    reward_sum += reward
                if game.is_planted and not planted:
                    planted = True
                    alive = [c for c in game.chars if c.team == "D" and c.is_alive]
                    plant_d = len(alive)
                    plant_a = sum(c.is_alive for c in game.chars if c.team == "A")
                    distance = scenario.rally_dist[scenario.site_of(game.planted_pos)]
                    plant_distance = float(np.mean([distance[tuple(c.pos)] for c in alive])) if alive else None
                    arrival = retake_arrival_readiness([distance[tuple(c.pos)] for c in alive], game.detonate_timer)
                    plant_reserve = retained_resources(alive, initial_charges)
                    if phase == "search":
                        for name, index in last_indices.items():
                            char = current[name]
                            bonus = .12 * plant_d - .04 * plant_a + PLANT_ADVANTAGE_REWARD * (plant_d - plant_a)
                            bonus -= arrival["plant_arrival_penalty"]
                            if char.is_alive:
                                bonus += .5 * np.exp(-max(0, distance[tuple(char.pos)]) / 8)
                                if initial_charges[name]:
                                    bonus += .3 * min(1., int(getattr(char, char.ability_name.lower() + "_charges", 0)) / initial_charges[name])
                            transitions[index][2] += bonus
                            transitions[index][5] = 1.
                            reward_sum += bonus
                        search_closed = True
                        break
            won = None if phase == "search" and planted else game.defender_wins > score_before
            if phase == "search" and planted and not game.round_over:
                # Record the observed site, without inventing a match result.
                controller.previous_rounds.append({"site": scenario.site_of(game.planted_pos), "winner": None})
            search_survivors = plant_d if planted else sum(c.is_alive for c in game.chars if c.team == "D")
            search_reserve = plant_reserve if planted else retained_resources([c for c in game.chars if c.team == "D"], initial_charges)
            readiness = .2 * search_survivors + .3 * (search_reserve if search_reserve is not None else 1.)
            if planted:
                readiness -= .02 * (plant_distance if plant_distance is not None else 60.)
                readiness += .05 * (5 - plant_a)
                readiness += PLANT_ADVANTAGE_REWARD * (plant_d - plant_a)
                readiness -= arrival["plant_arrival_penalty"]
            else:
                readiness += .5 if won else -1.
            if phase == "retake" or not search_closed:
                terminal_reward = (4. if won else -4.) if phase == "retake" else (3. if won else -3.)
                for index in last_indices.values():
                    transitions[index][2] += terminal_reward
                    transitions[index][5] = 1.
                    reward_sum += terminal_reward
            record = dict(opponent=opponent, preset=preset_name, round=round_no, won=won,
                          scope="search" if phase == "search" else "retake",
                          planted=planted, defused=bool(game.is_defused), plant_defenders=plant_d,
                          plant_attackers=plant_a, plant_distance=plant_distance,
                          plant_advantage=plant_d - plant_a if planted else None,
                          end_defenders=sum(c.is_alive for c in game.chars if c.team == "D"),
                          end_attackers=sum(c.is_alive for c in game.chars if c.team == "A"),
                          retake_ticks=retake_steps, reward=reward_sum,
                          site=scenario.site_of(game.planted_pos) if planted else None,
                          plant_resource_retention=plant_reserve, skill_uses=skill_uses,
                          search_survivors=search_survivors, search_resource_retention=search_reserve, search_readiness=readiness,
                          transitions=len(last_indices))
            rounds.append(record)
            record.update(arrival)
            if logger:
                logger.info("[%s][%s][%s] 相手AI=%s %s=%s R%02d %s plant=%s alive_at_plant=D%s/A%s 人数差=%s distance=%s reserve=%s skills=%d reward=%.2f",
                    phase, "学習" if training else "評価", log_context, opponent,
                    "学習編成" if training else "評価編成", preset_name, round_no,
                    "PLANTED" if phase == "search" and planted else "WIN" if won else "LOSS", planted,
                    plant_d, plant_a, "-" if not planted else f"{plant_d - plant_a:+d}",
                    "-" if plant_distance is None else f"{plant_distance:.1f}",
                    "-" if plant_reserve is None else f"{plant_reserve:.0%}", skill_uses, reward_sum)
                if planted:
                    logger.info("  リテイク参加余裕: 参加可能=%d/%d 遅延=%d 最遠=%.1ftick 最小余裕=%.1ftick 必要余裕=%dtick 減点=%.2f",
                                arrival["plant_ready_count"], plant_d, arrival["plant_late_count"],
                                arrival["plant_max_distance"] if plant_d else float("nan"),
                                arrival["plant_min_margin"] if plant_d else float("nan"),
                                RETAKE_SAFETY_MARGIN_TICKS, arrival["plant_arrival_penalty"])
            if round_callback:
                round_callback(record)
            if round_no < 12:
                game.current_round += 1
                game.init_round()
    return rounds, transitions


def summarize_defender(rounds):
    planted = [r for r in rounds if r["planted"]]
    finished = [r for r in rounds if r["won"] is not None]
    retakes = [r for r in planted if r.get("scope") != "search"]
    def average(rows, key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    return dict(rounds=len(rounds), plants=len(planted), completed_rounds=len(finished),
                scope="search" if rounds and all(r.get("scope") == "search" for r in rounds) else "retake",
                wins=sum(bool(r["won"]) for r in finished),
                win_rate=sum(bool(r["won"]) for r in finished) / len(finished) if finished else None,
                preplant_wins=sum(bool(r["won"]) and not r["planted"] for r in rounds),
                defuses=sum(r["defused"] for r in retakes),
                retake_win_rate=sum(bool(r["won"]) for r in retakes) / len(retakes) if retakes else None,
                mean_plant_defenders=average(planted, "plant_defenders"), mean_plant_attackers=average(planted, "plant_attackers"),
                mean_plant_advantage=average(planted, "plant_advantage"),
                all_ready_rate=average(planted, "plant_all_ready"),
                mean_plant_late_count=average(planted, "plant_late_count"),
                mean_plant_max_distance=average(planted, "plant_max_distance"),
                mean_plant_min_margin=average(planted, "plant_min_margin"),
                mean_plant_arrival_penalty=average(planted, "plant_arrival_penalty"),
                mean_plant_distance=average(planted, "plant_distance"), mean_retake_ticks=average(planted, "retake_ticks"),
                mean_resource_retention=average(planted, "plant_resource_retention"),
                mean_skill_uses=average(rounds, "skill_uses"),
                mean_search_survivors=average(rounds, "search_survivors"),
                mean_search_resources=average(rounds, "search_resource_retention"),
                mean_search_readiness=average(rounds, "search_readiness"),
                mean_reward=average(rounds, "reward"))


def evaluation_plan(opponents, presets, count, seed):
    return [(opponent, eligible_presets(presets, opponent)[i % len(eligible_presets(presets, opponent))],
             seed + 900000000 + list(OPPONENTS).index(opponent) * 1000000 + i * 100)
            for opponent in opponents for i in range(count)]


def evaluation_summary(metrics):
    def number(key, suffix="", digits=1):
        value = metrics.get(key)
        return "-" if value is None else f"{value:.{digits}f}{suffix}"
    def rate(key):
        value = metrics.get(key)
        return "-" if value is None else f"{value:.1%}"
    outcomes = (f"プラント前に決着={metrics['completed_rounds']} 勝利={metrics['preplant_wins']}" if metrics.get("scope") == "search" else
                f"勝率={rate('win_rate')} リテイク勝率={rate('retake_win_rate')} 解除={metrics['defuses']}")
    return (f"ラウンド={metrics['rounds']} プラント={metrics['plants']} {outcomes}\n"
            f"  プラント時の平均生存: 味方={number('mean_plant_defenders', '人', 2)} 相手={number('mean_plant_attackers', '人', 2)} "
            f"平均人数差={number('mean_plant_advantage', '人', 2)}\n"
            f"  a/b/c合流までの平均通路距離={number('mean_plant_distance', 'tick')} "
            f"スキル温存率={rate('mean_resource_retention')} 平均使用回数={number('mean_skill_uses', '回')}\n"
            f"  全生存者が余裕を持って参加可能={rate('all_ready_rate')} 平均遅延人数={number('mean_plant_late_count', '人', 2)} "
            f"最小余裕の平均={number('mean_plant_min_margin', 'tick')}\n"
            f"  search準備スコア={number('mean_search_readiness', digits=3)}")


def evaluate(plan, scenario, search, retake, analyses, phase, logger, log_context=""):
    rounds = []
    for opponent, preset, seed in plan:
        records, _ = rollout(opponent, preset, scenario, search, retake, analyses, phase, seed, logger=logger,
                             log_context=f"{log_context} seed={seed}")
        rounds.extend(records)
    metrics = summarize_defender(rounds)
    metrics["by_opponent"] = {opponent: summarize_defender([r for r in rounds if r["opponent"] == opponent])
                              for opponent in sorted({r["opponent"] for r in rounds})}
    return metrics


def score_best(metrics, phase):
    # A plant is permitted: survival, reserved utility and readiness matter.
    if phase == "search":
        return (metrics["mean_search_readiness"], metrics["mean_search_survivors"],
                metrics["mean_search_resources"], -(metrics["mean_plant_distance"] or 0.))
    return (metrics["retake_win_rate"] or 0., metrics["defuses"], metrics["mean_reward"] if metrics["mean_reward"] is not None else -1e9)


def atomic_save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("search", "retake"), default="search")
    p.add_argument("--sets", type=int, default=TRAINING_SETS)
    p.add_argument("--opponents", nargs="+", choices=tuple(OPPONENTS), default=list(OPPONENTS))
    p.add_argument("--train-presets", nargs="+", default=list(TRAINING_PRESETS))
    p.add_argument("--eval-presets", nargs="+", default=list(EVALUATION_PRESETS))
    p.add_argument("--eval-every", type=int, default=EVALUATION_INTERVAL)
    p.add_argument("--eval-seeds", type=int, default=EVALUATION_SEED_COUNT)
    p.add_argument("--updates", type=int, default=UPDATES_PER_BLOCK)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Resume latest checkpoint; default starts a new training run")
    mode.add_argument("--fresh", action="store_true", help="Start a new training run (same as the default)")
    mode.add_argument("--eval-only", action="store_true")
    p.add_argument("--data-dir", type=Path, default=HERE / "data" / "defender")
    p.add_argument("--best-dir", type=Path, default=DEFENDER_BEST)
    p.add_argument("--log-dir", type=Path, default=HERE / "logs" / "defender")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    if min(args.sets, args.eval_seeds, args.updates, args.batch_size) < 1 or args.eval_every < 0:
        raise ValueError("Positive training counts and nonnegative eval-every are required")
    if len(set(args.opponents)) != len(args.opponents):
        raise ValueError("Duplicate opponents")
    scenario = Scenario()
    for opponent in args.opponents:
        eligible_presets(args.train_presets, opponent)
        eligible_presets(args.eval_presets, opponent)
    analyses, hashes = load_analyses(scenario, args.opponents)
    if args.phase == "retake":
        from toruAI_v4.tv4_train_retake import train_retake
        return train_retake(args, scenario, analyses, hashes)
    for opponent in args.opponents:
        selected = argparse.Namespace(**vars(args))
        selected.opponents = [opponent]
        train_search_opponent(selected, scenario, {opponent: analyses[opponent]}, {opponent: hashes[opponent]})


def train_search_opponent(args, scenario, analyses, hashes):
    """Independent weights, replay, optimizer, evaluation and best per attacker."""
    opponent = args.opponents[0]
    schema = policy_metadata(scenario)
    torch.set_num_threads(1)
    seed_all(args.seed)
    latest = args.data_dir.resolve() / "search" / opponent / "latest.pt"
    best = args.best_dir.resolve() / opponent / "search_best.pt"
    log_dir = args.log_dir.resolve() / "search" / opponent
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("toru_v4_defender")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.StreamHandler(), logging.FileHandler(log_dir / "training.log", mode="w", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    replay = deque(maxlen=REPLAY_SIZE)
    model = DefenderDQN(OBS_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    rng = np.random.default_rng(args.seed)
    completed = 0
    search = model
    frozen_search_hash = None
    contract = dict(phase=args.phase, opponent=opponent, schema=schema, analysis_hashes=hashes, opponents=args.opponents,
                    training_presets=args.train_presets, evaluation_presets=args.eval_presets, seed=args.seed,
                    frozen_search_hash=frozen_search_hash)
    try:
        if args.resume:
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            saved = normalize_policy_schema(saved, scenario)
            if any(saved.get(k) != value for k, value in contract.items()):
                raise ValueError("Training checkpoint conditions changed; use a separate configuration or --fresh")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            completed = saved["completed_sets"]
            if saved.get("training_objective") == SEARCH_EVALUATION_SCOPE and saved.get("rally_geometry") == schema["staging"]:
                replay.extend([t[0].numpy(), t[1], t[2], t[3].numpy(), t[4].numpy(), t[5]] for t in saved["replay"])
            else:
                logger.info("合流地点の新報酬に切替: 重み・完了セット数は継続、旧報酬のreplayは引き継ぎません")
            rng.bit_generator.state = json.loads(saved["rng"])
        if args.eval_only:
            model, saved = load_policy(best, args.phase, scenario)
            if any(saved.get(k) != value for k, value in contract.items()):
                raise ValueError("Evaluation configuration differs from trained best")
            completed = saved["completed_sets"]
        target = copy.deepcopy(model)
        if args.resume and saved.get("training_objective") == SEARCH_EVALUATION_SCOPE and saved.get("target") is not None:
            target.load_state_dict(saved["target"])
        search, retake = model, None
        plan = evaluation_plan(args.opponents, args.eval_presets, args.eval_seeds, args.seed)
        comparison = None
        target_sets = completed + args.sets
        logger.info("相手別defender mode=%s phase=%s 相手AI=%s roster_shared=True sets=%d completed=%d 終了予定=%d",
                    "評価のみ" if args.eval_only else "再開" if args.resume else "新規", args.phase, opponent, args.sets, completed, target_sets)
        logger.info("学習編成=%s 評価編成=%s 評価=%dラウンド(%dseed/相手)", args.train_presets, args.eval_presets, len(plan) * 12, args.eval_seeds)
        logger.info("ログ=%s latest=%s best=%s", log_dir / "training.log", latest, best)
        logger.info("合流地点=L/Rマップのa・b・c 到着基準=起爆残りtick-突入交戦%d-解除6-余裕%d", RETAKE_COMBAT_RESERVE_TICKS, RETAKE_SAFETY_MARGIN_TICKS)
        for additional in range(1, (1 if args.eval_only else args.sets) + 1):
            if not args.eval_only:
                set_no = completed + 1
                for opponent in args.opponents:
                    block_started = time.perf_counter()
                    names = eligible_presets(args.train_presets, opponent)
                    preset = names[int(rng.integers(len(names)))]
                    epsilon = max(.03, .3 * np.exp(-set_no / 30))
                    demonstrations = max(0., 1 - (set_no - 1) / 5)
                    seed = args.seed + list(OPPONENTS).index(opponent) * 1000000 + set_no * 100
                    records, samples = rollout(opponent, preset, scenario, search, retake, analyses, args.phase,
                                               seed, training=True, epsilon=epsilon,
                                               teacher_probability=demonstrations, logger=logger,
                                               log_context=f"set={set_no}/{target_sets}")
                    replay.extend(samples)
                    loss = learn_dqn(model, target, optimizer, replay, rng, args.updates, args.batch_size)
                    logger.info("[%s][学習 set=%d/%d] 相手AI=%s 学習編成=%s サンプル=%d replay=%d loss=%s 探索率=%.3f 補助率=%.2f 所要=%.1f秒",
                                args.phase, set_no, target_sets, opponent, preset, len(samples), len(replay),
                                "-" if loss is None else f"{loss:.4f}", epsilon, demonstrations, time.perf_counter() - block_started)
                completed += 1
                state = {**contract, "training_objective": SEARCH_EVALUATION_SCOPE, "rally_geometry": schema["staging"], "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "target": target.state_dict(),
                         "completed_sets": completed, "rng": json.dumps(rng.bit_generator.state),
                         "replay": [[torch.from_numpy(t[0]), int(t[1]), float(t[2]), torch.from_numpy(t[3]), torch.from_numpy(t[4]), float(t[5])] for t in replay]}
                atomic_save(latest, state)
            if args.eval_only or (args.eval_every and completed % args.eval_every == 0) or additional == args.sets:
                metrics = evaluate(plan, scenario, search, retake, analyses, args.phase, logger, f"set={completed}/{target_sets}")
                logger.info("[%s][評価サマリ set=%d] %s", args.phase, completed, evaluation_summary(metrics))
                for opponent, values in metrics["by_opponent"].items():
                    logger.info("[%s][評価サマリ] 相手AI=%s %s", args.phase, opponent, evaluation_summary(values))
                if not args.eval_only:
                    if comparison is None and best.exists():
                        previous_model, previous = load_policy(best, args.phase, scenario)
                        if any(previous.get(k) != value for k, value in contract.items()):
                            raise ValueError("Best comparison conditions changed")
                        if previous.get("evaluation_plan") == plan and previous.get("evaluation_scope") == SEARCH_EVALUATION_SCOPE and previous.get("rally_geometry") == schema["staging"]:
                            comparison = previous["evaluation"]
                        else:
                            comparison = evaluate(plan, scenario, previous_model if args.phase == "search" else search,
                                                  previous_model if args.phase == "retake" else None, analyses, args.phase, logger)
                    usable = bool(metrics["plants"]) if args.phase == "retake" else True
                    if usable and (comparison is None or score_best(metrics, args.phase) > score_best(comparison, args.phase)):
                        atomic_save(best, {**contract, "model": model.state_dict(), "completed_sets": completed,
                                           "evaluation_scope": SEARCH_EVALUATION_SCOPE,
                                           "rally_geometry": schema["staging"],
                                           "evaluation_plan": plan, "evaluation": metrics})
                        comparison = metrics
                        logger.info("[%s][best] 保存/更新: %s", args.phase, best)
                    else:
                        logger.info("[%s][best] 維持 (retake評価でプラントなしの場合も更新しません)", args.phase)
        logger.info("完了 phase=%s completed_sets=%d 採用モデル=%s", args.phase, completed, best)
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
