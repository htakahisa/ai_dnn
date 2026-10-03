"""Train ConCon attacker routes until planting or round end (or in the legacy simulator)."""

import argparse
import io
import random
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1.co1_attacker_common import (
    ACTION_DIM, ACTION_PLANT, ACTION_WAIT, ATTACKER_SPAWNS, CARDINAL_MOVES,
    DEFAULT_SAVE_DIR, GAME_MAZE_STR, GORIGONS, GRID, HEIGHT, LEFT_PLANT_CELLS,
    MAX_CANDIDATE_BFS_DISTANCE, MAX_TICKS, OBS_DIM, PLANT_REQUIRED_TICKS,
    SPIKE_CARRIER_INDEX, SPLIT_PATTERNS, STRATEGY_MAZE_STR, WAYPOINT_ORDER,
    WAYPOINT_POINTS, WIDTH, RouteProgress, SharedRouteDQN, advance_team_routes,
    bfs_distance_map, build_action_mask, build_observation, build_team_route_action_mask, choose_split_assignment,
    choose_team_fire_target, facing_for_fire_target, parse_game_grid,
    parse_strategy_points, plant_stage_action_mask, select_nearest_candidate, _choose_action,
)
from concon_v1.co1_attacker_scenarios import (
    SCENARIOS, get_scenario, validate_checkpoint_scenario,
)

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

TARGET_UPDATE_INTERVAL = 1000
GAMMA = 0.99

DEFAULT_EPISODES = 1000
CHECKPOINT_INTERVAL = 50  # bestモデル算出episode間隔
DEFAULT_EVAL_ROUNDS = 20  # 探索なし評価の各相手teamとの試合数
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_RATIO = 0.7
FORCE_SAVE = True  # Keep numbered debug models at every checkpoint interval.


def epsilon_by_episode(episode, total_episodes=DEFAULT_EPISODES):
    decay_episodes = max(1, int(total_episodes * EPSILON_DECAY_RATIO))
    fraction = min(max(float(episode) / decay_episodes, 0.0), 1.0)
    if fraction >= 1.0:
        return EPSILON_END
    return EPSILON_START + (EPSILON_END - EPSILON_START) * fraction


class RouteEnv:
    def __init__(self, seed=None, map_name="A1"):
        self.scenario = get_scenario(map_name)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.pattern_index, groups = choose_split_assignment(
            self.rng, a_point_count=len(self.scenario.waypoint_points["a"]),
            balanced_only="a" in self.scenario.uppercase_markers,
        )
        self.positions = list(self.scenario.attacker_spawns)
        self.routes = [
            RouteProgress(group, self.pattern_index, pos, scenario=self.scenario)
            for group, pos in zip(groups, self.positions)
        ]
        self.alive = [True] * len(self.positions)
        self._a_completed_groups = set()
        self.plant_progress = 0
        self.elapsed_ticks = 0
        self.done = False
        self.success = False
        return self._collect()

    def _collect(self):
        observations = []
        masks = []
        for index, (position, route) in enumerate(zip(self.positions, self.routes)):
            allies = [other for other_index, other in enumerate(self.positions) if other_index != index]
            observations.append(build_observation(
                route, position, index == SPIKE_CARRIER_INDEX, allies,
                self.plant_progress if index == SPIKE_CARRIER_INDEX else 0, self.elapsed_ticks,
            ))
            masks.append(build_team_route_action_mask(
                self.routes, self.positions, self.alive, index, self.scenario.grid,
                SPIKE_CARRIER_INDEX, self._a_completed_groups,
            ))
        return observations, masks

    def _advance_routes_if_reached(self):
        advance_team_routes(self.routes, self.positions, self.alive, self._a_completed_groups,
                            rng=self.rng)

    def step(self, actions):
        observations, masks = self._collect()
        previous_stages = [route.stage for route in self.routes]
        previous_distances = [
            int(route.distance_map[position])
            for route, position in zip(self.routes, self.positions)
        ]
        rewards = [-0.005] * len(self.positions)

        for index, action in enumerate(actions):
            if not masks[index][int(action)]:
                action = ACTION_WAIT
            row, col = self.positions[index]
            if action < len(CARDINAL_MOVES):
                row_delta, col_delta = CARDINAL_MOVES[action]
                destination = (row + row_delta, col + col_delta)
                if destination not in self.positions:
                    self.positions[index] = destination
                if index == SPIKE_CARRIER_INDEX and self.positions[index] != (row, col):
                    self.plant_progress = 0
            elif action == ACTION_PLANT and index == SPIKE_CARRIER_INDEX:
                self.plant_progress += 1
                rewards[index] += 0.05
                if self.plant_progress >= PLANT_REQUIRED_TICKS:
                    self.done = True
                    self.success = True
            elif index == SPIKE_CARRIER_INDEX:
                self.plant_progress = 0

        self._advance_routes_if_reached()

        self.elapsed_ticks += 1
        for index, (route, position) in enumerate(zip(self.routes, self.positions)):
            if route.stage != previous_stages[index]:
                rewards[index] += 0.25
            elif previous_distances[index] >= 0:
                distance = int(route.distance_map[position])
                if distance >= 0:
                    rewards[index] += 0.04 * (previous_distances[index] - distance)

        if self.success:
            rewards = [reward + 10.0 for reward in rewards]
        elif self.elapsed_ticks >= MAX_TICKS:
            self.done = True
            rewards = [reward - 3.0 for reward in rewards]

        next_observations, next_masks = self._collect()
        return observations, masks, rewards, next_observations, next_masks, self.done


class RouteReplayCollector:
    """Keep each decision open until that actor decides again or its phase ends.

    Contact/ability ticks belong to the preceding route decision. Bootstrap at
    the next actual policy input, discounting by the number of elapsed ticks.
    """

    def __init__(self, player_count, gamma=GAMMA):
        self.gamma = gamma
        self.pending = [None] * player_count

    def _finish(self, index, observation, mask, terminal, replay):
        pending = self.pending[index]
        if pending is None:
            return
        replay.append((pending["observation"], pending["action"], pending["reward"],
                       observation.copy(), mask.copy(), float(terminal), pending["ticks"]))
        self.pending[index] = None

    def record(self, transition, actions, policy_applied, active_before, alive,
               replay, *, route_active=True, interrupted=False):
        old_obs, old_masks, rewards, next_obs, next_masks, done = transition
        if not route_active:
            return 0.0
        total_reward = 0.0
        for index, action in enumerate(actions):
            if not active_before[index]:
                continue
            if policy_applied[index]:
                self._finish(index, old_obs[index], old_masks[index], False, replay)
                self.pending[index] = {
                    "observation": old_obs[index].copy(), "action": int(action),
                    "reward": 0.0, "ticks": 0,
                }
            pending = self.pending[index]
            if pending is not None:
                pending["reward"] += self.gamma ** pending["ticks"] * rewards[index]
                pending["ticks"] += 1
                total_reward += rewards[index]
            if done or interrupted or not alive[index]:
                self._finish(index, next_obs[index], next_masks[index], True, replay)
        return total_reward


def _optimize(model, target, optimizer, replay, batch_size, gamma):
    if len(replay) < batch_size:
        return
    batch = random.sample(replay, batch_size)
    observations, actions, rewards, next_observations, next_masks, dones, durations = zip(*batch)
    observations = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.int64)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_observations = torch.as_tensor(np.asarray(next_observations), dtype=torch.float32)
    next_masks = torch.as_tensor(np.asarray(next_masks), dtype=torch.bool)
    dones = torch.as_tensor(dones, dtype=torch.float32)
    durations = torch.as_tensor(durations, dtype=torch.float32)

    selected = model(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_policy = model(next_observations).masked_fill(~next_masks, -torch.inf)
        next_actions = next_policy.argmax(dim=1)
        next_values = target(next_observations).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected = rewards + gamma ** durations * next_values * (1.0 - dones)
    loss = nn.functional.smooth_l1_loss(selected, expected)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()


def summarize_team_plants(opponents, results):
    """Count plants and played episodes for each sampled opponent."""
    summary = {name: {"plants": 0, "episodes": 0} for name in dict.fromkeys(opponents)}
    for name, planted in results:
        summary[name]["episodes"] += 1
        summary[name]["plants"] += int(planted)
    for counts in summary.values():
        games = counts["episodes"]
        counts["plant_rate"] = counts["plants"] / games if games else None
    return summary


def format_team_plants(summary):
    parts = []
    for name, counts in summary.items():
        rate = counts["plant_rate"]
        rate_text = f"{rate:.3f}" if rate is not None else "-"
        parts.append(f"{name}={counts['plants']}/{counts['episodes']}({rate_text})")
    return " ".join(parts)


def summarize_team_rounds(opponents, results):
    """Aggregate elapsed ticks and actual timeouts separately for each opponent."""
    summary = {name: {"episodes": 0, "ticks": 0, "timeouts": 0}
               for name in dict.fromkeys(opponents)}
    for name, ticks, timed_out in results:
        counts = summary[name]
        counts["episodes"] += 1
        counts["ticks"] += ticks
        counts["timeouts"] += int(timed_out)
    for counts in summary.values():
        counts["avg_ticks"] = (counts["ticks"] / counts["episodes"]
                               if counts["episodes"] else None)
    return summary


def format_team_round_metric(summary, recent_summary, metric):
    def format_value(value):
        if value is None:
            return "-"
        return f"{value:.2f}" if metric == "avg_ticks" else str(value)

    return " ".join(
        f"{name}={format_value(counts[metric])}"
        f"(recent100={format_value(recent_summary[name][metric])})"
        for name, counts in summary.items()
    )


def evaluate_checkpoint(checkpoint, rounds=DEFAULT_EVAL_ROUNDS, seed=0):
    """Evaluate frozen weights greedily without changing training RNG streams."""
    if rounds < 1:
        raise ValueError("evaluation rounds must be positive")
    scenario = get_scenario(checkpoint.get("map_name", "A1"))
    validate_checkpoint_scenario(checkpoint, scenario)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        if checkpoint["training_mode"] == "battle":
            from concon_v1.evaluate_co1_attacker import evaluate

            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            frozen_checkpoint = buffer.getvalue()
            summary = {}
            for opponent in dict.fromkeys(checkpoint["opponents"]):
                result = evaluate(opponent, rounds, seed,
                                  frozen_checkpoint=frozen_checkpoint, map_name=scenario.map_name)
                summary[opponent] = {
                    "plants": result["plants"],
                    "episodes": result["rounds"],
                    "plant_rate": result["plant_success_rate"],
                }
                print("  eval_team " + format_team_plants({opponent: summary[opponent]}),
                      flush=True)
            rates = [counts["plant_rate"] for counts in summary.values()]
            success_rate = sum(rates) / len(rates)
            minimum_rate = min(rates)
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = SharedRouteDQN(obs_dim=scenario.obs_dim)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            env = RouteEnv(seed, map_name=scenario)
            rng = random.Random(seed)
            plants = 0
            for _ in range(rounds):
                observations, masks = env.reset()
                while not env.done:
                    actions = [_choose_action(model, obs, mask, 0.0, rng)
                               for obs, mask in zip(observations, masks)]
                    _, _, _, observations, masks, _ = env.step(actions)
                plants += int(env.success)
            summary = {}
            success_rate = minimum_rate = plants / rounds
        return {
            "success_rate": success_rate,
            "min_team_plant_rate": minimum_rate,
            "team_plants": summary,
            "rounds_per_opponent": rounds,
            "seed": seed,
            "epsilon": 0.0,
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def qualifies_as_best(evaluation, best_evaluation):
    """Keep both the mean and the weakest opponent rate from regressing."""
    return best_evaluation is None or (
        evaluation["success_rate"] >= best_evaluation["success_rate"]
        and evaluation["min_team_plant_rate"] >= best_evaluation["min_team_plant_rate"]
    )


def train(episodes=DEFAULT_EPISODES, save_dir=None, seed=0,
          mode="battle", opponents=None, eval_rounds=DEFAULT_EVAL_ROUNDS, map_name="A1",
          force_save=FORCE_SAVE):
    if eval_rounds < 1:
        raise ValueError("evaluation rounds must be positive")
    scenario = get_scenario(map_name)
    save_dir = Path(save_dir) if save_dir is not None else scenario.save_dir
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SharedRouteDQN(obs_dim=scenario.obs_dim)
    target = SharedRouteDQN(obs_dim=scenario.obs_dim)
    target.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    replay = deque(maxlen=100_000)
    if mode == "battle":
        from concon_v1.co1_battle_training import BattleRouteEnv
        env = BattleRouteEnv(seed, opponents, model=model, map_name=scenario, learn_setup=True)
    elif mode == "route":
        env = RouteEnv(seed, map_name=scenario)
    else:
        raise ValueError("mode must be 'battle' or 'route'")
    global_step = 0
    recent_success = deque(maxlen=100)
    recent_ticks = deque(maxlen=100)
    recent_timeouts = deque(maxlen=100)
    recent_drops = deque(maxlen=100)
    recent_recoveries = deque(maxlen=100)
    total_plants = 0
    total_ticks = 0
    total_timeouts = 0
    team_results = []
    recent_team_results = deque(maxlen=100)
    recent_team_wins = deque(maxlen=100)
    team_round_results = []
    recent_team_round_results = deque(maxlen=100)
    best_evaluation = None
    checked_existing_best = False
    started = time.perf_counter()

    for episode in range(1, episodes + 1):
        observations, masks = env.reset()
        route_replay = RouteReplayCollector(len(observations))
        epsilon = epsilon_by_episode(episode, episodes)
        total_reward = 0.0
        while not env.done:
            active_before = list(env.alive)
            if mode == "battle":
                transition = env.step(epsilon=epsilon, action_rng=rng)
                actions = env.actions
            else:
                actions = [
                    _choose_action(model, observations[index], masks[index], epsilon, rng,
                                   route=env.routes[index], position=env.positions[index])
                    for index in range(len(observations))
                ]
                transition = env.step(actions)
            old_obs, old_masks, rewards, next_obs, next_masks, done = transition
            route_active = mode != "battle" or env.route_active_before_step
            route_interrupted = mode == "battle" and (env.retrieve_active or env.success)
            if mode == "battle":
                total_reward += route_replay.record(
                    transition, env.actions, env.policy_action_applied, active_before,
                    env.alive, replay, route_active=route_active,
                    interrupted=route_interrupted,
                )
            else:
                for index, action in enumerate(actions):
                    replay.append((old_obs[index], action, rewards[index], next_obs[index],
                                   next_masks[index], float(done), 1))
                    total_reward += rewards[index]
            observations, masks = next_obs, next_masks
            if route_active:
                global_step += 1
                _optimize(model, target, optimizer, replay, 128, GAMMA)
                if global_step % TARGET_UPDATE_INTERVAL == 0:
                    target.load_state_dict(model.state_dict())

        planted = bool(env.success)
        total_plants += int(planted)
        recent_success.append(float(planted))
        total_ticks += env.elapsed_ticks
        recent_ticks.append(env.elapsed_ticks)
        timed_out = not planted and (
            env.game.round_over and env.game.round_timer <= 0
            if mode == "battle" else env.elapsed_ticks >= MAX_TICKS
        )
        total_timeouts += int(timed_out)
        recent_timeouts.append(int(timed_out))
        if mode == "battle":
            result = (env.opponent, planted)
            team_results.append(result)
            recent_team_results.append(result)
            # Carry training ends at planting, before a possible defuse.
            # Count that objective plus actual pre-plant elimination wins.
            carry_success = planted or bool(env.game.round_over and env.game.attacker_wins)
            recent_team_wins.append((env.opponent, carry_success))
            round_result = (env.opponent, env.elapsed_ticks, timed_out)
            team_round_results.append(round_result)
            recent_team_round_results.append(round_result)
            recent_drops.append(int(env.had_spike_drop))
            recent_recoveries.append(int(env.spike_recovered))
        success_rate = sum(recent_success) / len(recent_success)
        plant_rate_total = total_plants / episode
        if episode % 20 == 0:
            recovery = (f" recovered100={sum(recent_recoveries)}/{sum(recent_drops)}"
                        if mode == "battle" else "")
            print(
                f"episode={episode}/{episodes} success100={success_rate:.3f}"
                f" plant_total={total_plants}/{episode}"
                f" plant_rate_total={plant_rate_total:.3f}"
                f"{recovery} reward={total_reward:.2f} ticks={env.elapsed_ticks} "
                f"epsilon={epsilon:.3f} elapsed={time.perf_counter() - started:.1f}s"
            )
            if mode == "battle":
                print("  team_total " + format_team_plants(
                    summarize_team_plants(env.opponents, team_results)))
                print(f"{GREEN}  team100     " + format_team_plants(
                    summarize_team_plants(env.opponents, recent_team_results)) + f"{RESET}")
                print(f"{YELLOW}  team_win100 " + format_team_plants(
                    summarize_team_plants(env.opponents, recent_team_wins))
                    + " (plant_or_elimination)" + f"{RESET}")
                round_summary = summarize_team_rounds(env.opponents, team_round_results)
                recent_round_summary = summarize_team_rounds(env.opponents, recent_team_round_results)
                print("  team_avg_ticks " + format_team_round_metric(
                    round_summary, recent_round_summary, "avg_ticks"))
                print("  team_timeouts " + format_team_round_metric(
                    round_summary, recent_round_summary, "timeouts"))
            else:
                print(f"  avg_ticks_total={total_ticks / episode:.2f}"
                      f" avg_ticks100={sum(recent_ticks) / len(recent_ticks):.2f}")
                print(f"  timeout_total={total_timeouts}"
                      f" timeout100={sum(recent_timeouts)}")
        if episode % CHECKPOINT_INTERVAL == 0 or episode == episodes:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "obs_dim": scenario.obs_dim,
                "n_actions": ACTION_DIM,
                "episode": episode,
                "success_rate": success_rate,
                "epsilon": epsilon,
                "epsilon_end": EPSILON_END,
                "best_selection": "greedy_mean_and_min_team",
                "evaluation": None,
                "plant_count_total": total_plants,
                "plant_rate_total": plant_rate_total,
                "team_win100": (summarize_team_plants(env.opponents, recent_team_wins)
                                if mode == "battle" else {}),
                "team_win100_definition": "plant_or_elimination",
                "split_patterns": SPLIT_PATTERNS,
                "map_name": scenario.map_name,
                "scenario_signature": scenario.signature,
                "waypoint_points": scenario.waypoint_points,
                "waypoint_order": scenario.waypoint_order,
                "plant_cells": scenario.plant_cells,
                "plant_side": scenario.plant_side,
                "max_candidate_bfs_distance": scenario.max_candidate_bfs_distance,
                "training_roster": GORIGONS.players,
                "spike_carrier": GORIGONS.spike_holder,
                "training_mode": mode,
                "attacker_perception": "iq" if mode == "battle" else "route_simulator",
                "opponents": env.opponents if mode == "battle" else (),
                "team_plant_total": (
                    summarize_team_plants(env.opponents, team_results)
                    if mode == "battle" else {}
                ),
                "team_plant100": (
                    summarize_team_plants(env.opponents, recent_team_results)
                    if mode == "battle" else {}
                ),
                "team_round_total": (
                    summarize_team_rounds(env.opponents, team_round_results)
                    if mode == "battle" else {}
                ),
                "team_round100": (
                    summarize_team_rounds(env.opponents, recent_team_round_results)
                    if mode == "battle" else {}
                ),
            }
            if scenario.map_name == "A1":
                checkpoint["left_plant_cells"] = scenario.plant_cells
            if force_save:
                debug_path = save_dir / f"co1_attacker_{scenario.map_name}_episode_{episode}.pt"
                torch.save(checkpoint, debug_path)
                print(f"Force-saved debug model at episode {episode} "
                      f"epsilon={epsilon:.3f}: {debug_path}", flush=True)
            if epsilon <= EPSILON_END:
                best_path = save_dir / scenario.checkpoint_filename("best")
                if not checked_existing_best:
                    if best_path.is_file():
                        previous = torch.load(best_path, map_location="cpu", weights_only=False)
                        try:
                            validate_checkpoint_scenario(previous, scenario)
                        except ValueError as error:
                            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            backup_path = best_path.with_name(
                                f"{best_path.stem}_incompatible_{stamp}{best_path.suffix}")
                            shutil.copy2(best_path, backup_path)
                            print(f"Skipping incompatible existing best: {error}. "
                                  f"Backed up to {backup_path}", flush=True)
                        else:
                            print("Evaluating existing best with the current opponents, "
                                  f"rounds={eval_rounds}/team seed={seed}", flush=True)
                            # Older best files only contain training success100. Re-evaluate
                            # their weights under the same conditions as the candidate.
                            previous["training_mode"] = mode
                            previous["opponents"] = checkpoint["opponents"]
                            best_evaluation = evaluate_checkpoint(previous, eval_rounds, seed)
                    checked_existing_best = True
                print(f"Evaluating episode {episode} with epsilon=0 "
                      f"rounds={eval_rounds}/team seed={seed}", flush=True)
                checkpoint["evaluation"] = evaluate_checkpoint(checkpoint, eval_rounds, seed)
            torch.save(checkpoint, save_dir / scenario.checkpoint_filename("latest"))
            evaluation = checkpoint["evaluation"]
            if evaluation is not None and qualifies_as_best(evaluation, best_evaluation):
                best_evaluation = evaluation
                torch.save(checkpoint, save_dir / scenario.checkpoint_filename("best"))
                print(
                    f"Saved best model with eval_mean={evaluation['success_rate']:.3f} "
                    f"eval_min_team={evaluation['min_team_plant_rate']:.3f} "
                    f"at episode {episode}", flush=True,
                )
            elif evaluation is not None:
                print(
                    f"Kept best: eval_mean={evaluation['success_rate']:.3f} "
                    f"(best={best_evaluation['success_rate']:.3f}) "
                    f"eval_min_team={evaluation['min_team_plant_rate']:.3f} "
                    f"(best={best_evaluation['min_team_plant_rate']:.3f})", flush=True,
                )
            else:
                print(f"Saved latest at episode {episode}; best evaluation starts "
                      f"when epsilon reaches {EPSILON_END:g} (current={epsilon:.3f})")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-map", "--map", dest="map_name", choices=SCENARIOS, default="A1")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-rounds", type=int, default=DEFAULT_EVAL_ROUNDS,
                        help="greedy evaluation games per opponent after epsilon reaches its floor")
    parser.add_argument("--save-dir", type=Path, help="override the selected map's model directory")
    parser.add_argument("--force-save", action="store_true", default=FORCE_SAVE,
                        help="keep numbered debug models at every checkpoint interval regardless of epsilon")
    parser.add_argument("--mode", choices=("battle", "route"), default="battle")
    parser.add_argument("--opponents", nargs="+", choices=(
        "omoko_v1", "touyama_v2", "fnatic_v3", "gc_v1", "toru_ai_v3.1",
    ))
    args = parser.parse_args()
    if args.eval_rounds < 1:
        parser.error("--eval-rounds must be positive")
    train(args.episodes, args.save_dir, args.seed, args.mode, args.opponents, args.eval_rounds,
          map_name=args.map_name, force_save=args.force_save)


if __name__ == "__main__":
    main()
