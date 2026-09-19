"""Seeded policy evaluation; metrics refer to the GC training simulators."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def evaluation_seed(seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def greedy_actions(model, observations, masks):
    names = list(observations)
    if not names:
        return {}
    device = next(model.parameters()).device
    with torch.no_grad():
        obs = torch.as_tensor(np.stack([observations[n] for n in names]), device=device)
        mask = torch.as_tensor(np.stack([masks[n] for n in names]), device=device)
        actions = model(obs).masked_fill(~mask, -float("inf")).argmax(dim=1).cpu().tolist()
    return dict(zip(names, actions))


def evaluate_guard(training, model, episodes=40, seed=20260917, positioning_version=2,
                   start_mode=None):
    per_pattern = defaultdict(Counter)
    total = Counter()
    with evaluation_seed(seed):
        env = training.GuardEnv(start_mode=start_mode)
        for ep in range(episodes):
            # Each initial scenario is independent of the previous policy's
            # episode length and number of random combat draws.
            random.seed(seed + ep)
            np.random.seed(seed + ep)
            marker = training.ACTIVE_GUARD_PATTERNS[ep % len(training.ACTIVE_GUARD_PATTERNS)]
            env.start_mode = start_mode or ("transition", "dispersed", "hold")[
                (ep // len(training.ACTIVE_GUARD_PATTERNS)) % 3]
            obs, masks = env.reset(pattern_marker=marker)
            stats = per_pattern[marker]
            stats["episodes"] += 1
            eligible = {a.name for a in env.attackers
                        if a.assigned_guard_dist_map[tuple(a.pos)] > 0}
            arrived = set()
            stats["off_position_starts"] += len(eligible)
            reward_sum = 0.0
            for _ in range(training.MAX_TICKS):
                if positioning_version == 0:
                    for name, observation in obs.items():
                        observation[33] = 0.0
                        observation[13] = 0.0
                        observation[32] = float(observation[29] <= 1.01 / (training.HEIGHT + training.WIDTH))
                if positioning_version < 2:
                    for name, observation in obs.items():
                        a = next(a for a in env.attackers if a.name == name)
                        observation[17] = float(any(training.has_los(a.pos, cell, env._smoke_cells())
                            for cell in training.postplant_watch_cells(training.GRID, env.planted_pos)))
                actions = greedy_actions(model, obs, masks)
                obs, masks, rewards, done = env.step(actions)
                reward_sum += sum(rewards.values())
                for a in env.attackers:
                    if not a.is_alive:
                        continue
                    distance = int(a.assigned_guard_dist_map[tuple(a.pos)])
                    stats["alive_ticks"] += 1
                    stats["guard_distance_sum"] += max(distance, 0)
                    if distance == 0:
                        stats["position_ticks"] += 1
                        if a.name in eligible:
                            arrived.add(a.name)
                if done or not obs:
                    break
            reason = env.match_over_reason or "unknown"
            stats[reason] += 1
            stats["wins"] += int(reason in ("attacker_win_wipe", "attacker_win_detonate"))
            stats["arrivals"] += len(arrived)
            stats["reward_sum"] += reward_sum
        for stats in per_pattern.values():
            total.update(stats)

    def summarize(stats):
        return {"episodes": stats["episodes"],
                "win_rate": stats["wins"] / max(1, stats["episodes"]),
                "arrival_rate": stats["arrivals"] / max(1, stats["off_position_starts"]),
                "off_position_starts": stats["off_position_starts"],
                "position_tick_rate": stats["position_ticks"] / max(1, stats["alive_ticks"]),
                "avg_guard_distance": stats["guard_distance_sum"] / max(1, stats["alive_ticks"]),
                "avg_reward": stats["reward_sum"] / max(1, stats["episodes"]),
                "defused": stats["defused"]}
    return {**summarize(total), "per_pattern": {str(p): summarize(s) for p, s in per_pattern.items()}}


def evaluate_carry(training, model, episodes=40, seed=20260917, priority_cells=None):
    stats = Counter()
    per_pattern = defaultdict(Counter)
    priority_map = None
    if priority_cells is not None:
        priority_map = training.bfs_distance_map_multi(priority_cells or training.PLANT_CELLS)
        finite = priority_map[priority_map >= 0]
        priority_max = int(finite.max()) if finite.size else training.HEIGHT + training.WIDTH
    with evaluation_seed(seed):
        env = training.CarryEnv()
        env.set_training_progress(1.0)
        for ep in range(episodes):
            random.seed(seed + ep)
            np.random.seed(seed + ep)
            marker = list(training.PLANT_PATTERNS)[ep % len(training.PLANT_PATTERNS)]
            obs, mask = env.reset(pattern_marker=marker)
            reward_sum = 0.0
            for _ in range(training.MAX_TICKS):
                if priority_map is not None:
                    pos = tuple(env.carrier.pos)
                    d = int(priority_map[pos])
                    obs[25] = min(d if d >= 0 else priority_max, priority_max) / max(1, priority_max)
                    obs[26], obs[27] = training.bfs_best_direction(priority_map, *pos)
                action = greedy_actions(model, {"carrier": obs}, {"carrier": mask})["carrier"]
                obs, mask, reward, done = env.step(action)
                reward_sum += reward
                if done:
                    break
            result = per_pattern[marker]
            result["episodes"] += 1
            planted = env.match_over_reason == "planted"
            result["plants"] += int(planted)
            result["preferred_plants"] += int(planted and tuple(env.carrier.pos) in training.PRIORITY_CELLS)
            result["target_plants"] += int(planted and tuple(env.carrier.pos) == env.target_plant_pos)
            result["reward_sum"] += reward_sum
        for result in per_pattern.values():
            stats.update(result)

    def summarize(result):
        n = max(1, result["episodes"])
        return {"episodes": result["episodes"], "plant_rate": result["plants"] / n,
                "preferred_plant_rate": result["preferred_plants"] / n,
                "target_plant_rate": result["target_plants"] / n,
                "avg_reward": result["reward_sum"] / n}
    return {**summarize(stats), "per_pattern": {str(p): summarize(s) for p, s in per_pattern.items()}}
