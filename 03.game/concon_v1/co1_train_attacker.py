"""Train ConCon attacker routes in real 5v5 rounds (or the legacy route-only simulator)."""

import argparse
import io
import random
import sys
import time
from collections import deque
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
    bfs_distance_map, build_action_mask, build_observation, choose_split_assignment,
    choose_team_fire_target, facing_for_fire_target, parse_game_grid,
    parse_strategy_points, plant_stage_action_mask, select_nearest_candidate, _choose_action,
)
from concon_v1.co1_attacker_scenarios import (
    SCENARIOS, get_scenario, validate_checkpoint_scenario,
)


TARGET_UPDATE_INTERVAL = 1000

DEFAULT_EPISODES = 2000
CHECKPOINT_INTERVAL = 50  # bestモデル算出episode間隔
DEFAULT_EVAL_ROUNDS = 20  # 探索なし評価の各相手teamとの試合数
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_RATIO = 0.7


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
            masks.append(build_action_mask(
                self.scenario.grid, position, allies, index == SPIKE_CARRIER_INDEX,
                route.at_plant_stage, route.goal,
                route.distance_map,
                route.stage == 0 and position == route.goal
                and bool({self.routes[i].group for i, alive in enumerate(self.alive) if alive}
                         - self._a_completed_groups),
            ))
            if route.at_plant_stage and index != SPIKE_CARRIER_INDEX:
                masks[-1] = plant_stage_action_mask(
                    self.scenario.grid, position, allies, self.positions[SPIKE_CARRIER_INDEX],
                    self.routes[SPIKE_CARRIER_INDEX].goal,
                )
        return observations, masks

    def _advance_routes_if_reached(self):
        advance_team_routes(self.routes, self.positions, self.alive, self._a_completed_groups)

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


def _optimize(model, target, optimizer, replay, batch_size, gamma):
    if len(replay) < batch_size:
        return
    batch = random.sample(replay, batch_size)
    observations, actions, rewards, next_observations, next_masks, dones = zip(*batch)
    observations = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.int64)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_observations = torch.as_tensor(np.asarray(next_observations), dtype=torch.float32)
    next_masks = torch.as_tensor(np.asarray(next_masks), dtype=torch.bool)
    dones = torch.as_tensor(dones, dtype=torch.float32)

    selected = model(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_policy = model(next_observations).masked_fill(~next_masks, -torch.inf)
        next_actions = next_policy.argmax(dim=1)
        next_values = target(next_observations).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected = rewards + gamma * next_values * (1.0 - dones)
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
          mode="battle", opponents=None, eval_rounds=DEFAULT_EVAL_ROUNDS, map_name="A1"):
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
        env = BattleRouteEnv(seed, opponents, model=model, map_name=scenario)
    elif mode == "route":
        env = RouteEnv(seed, map_name=scenario)
    else:
        raise ValueError("mode must be 'battle' or 'route'")
    global_step = 0
    recent_success = deque(maxlen=100)
    recent_drops = deque(maxlen=100)
    recent_recoveries = deque(maxlen=100)
    total_plants = 0
    team_results = []
    recent_team_results = deque(maxlen=100)
    best_evaluation = None
    checked_existing_best = False
    started = time.perf_counter()

    for episode in range(1, episodes + 1):
        observations, masks = env.reset()
        epsilon = epsilon_by_episode(episode, episodes)
        total_reward = 0.0
        while not env.done:
            active_before = list(env.alive)
            if mode == "battle":
                transition = env.step(epsilon=epsilon, action_rng=rng)
                actions = env.actions
            else:
                actions = [
                    _choose_action(model, observations[index], masks[index], epsilon, rng)
                    for index in range(len(observations))
                ]
                transition = env.step(actions)
            old_obs, old_masks, rewards, next_obs, next_masks, done = transition
            route_active = mode != "battle" or env.route_active_before_step
            route_interrupted = mode == "battle" and (env.retrieve_active or env.success)
            for index, action in enumerate(actions):
                if (not active_before[index] or not route_active
                        or (mode == "battle" and not env.policy_action_applied[index])):
                    continue
                # Retrieval is controlled by its own phase. A dropped spike
                # ends this route transition, but the real round continues.
                applied_action = env.actions[index] if mode == "battle" else action
                replay.append((old_obs[index], applied_action, rewards[index], next_obs[index],
                               next_masks[index], float(done or route_interrupted
                                                        or not env.alive[index])))
                total_reward += rewards[index]
            observations, masks = next_obs, next_masks
            if route_active:
                global_step += 1
                _optimize(model, target, optimizer, replay, 128, 0.99)
                if global_step % TARGET_UPDATE_INTERVAL == 0:
                    target.load_state_dict(model.state_dict())

        planted = bool(env.success)
        total_plants += int(planted)
        recent_success.append(float(planted))
        if mode == "battle":
            result = (env.opponent, planted)
            team_results.append(result)
            recent_team_results.append(result)
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
                print("  team100 " + format_team_plants(
                    summarize_team_plants(env.opponents, recent_team_results)))
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
            }
            if scenario.map_name == "A1":
                checkpoint["left_plant_cells"] = scenario.plant_cells
            if epsilon <= EPSILON_END:
                best_path = save_dir / scenario.checkpoint_filename("best")
                if not checked_existing_best:
                    if best_path.is_file():
                        print("Evaluating existing best with the current opponents, "
                              f"rounds={eval_rounds}/team seed={seed}", flush=True)
                        previous = torch.load(best_path, map_location="cpu", weights_only=False)
                        validate_checkpoint_scenario(previous, scenario)
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
    parser.add_argument("--mode", choices=("battle", "route"), default="battle")
    parser.add_argument("--opponents", nargs="+", choices=(
        "omoko_v1", "touyama_v2", "fnatic_v3", "gc_v1", "toru_ai_v3.1",
    ))
    args = parser.parse_args()
    if args.eval_rounds < 1:
        parser.error("--eval-rounds must be positive")
    train(args.episodes, args.save_dir, args.seed, args.mode, args.opponents, args.eval_rounds,
          map_name=args.map_name)


if __name__ == "__main__":
    main()
