"""Train one site/pattern guard model using the common postplant environment."""

import argparse
from collections import deque
import io
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from concon_v1.co1_guard_common import GuardDQN, ACTION_DIM, observation_dim, GORIGONS
from concon_v1.co1_guard_scenarios import SCENARIOS, get_scenario, validate_checkpoint
from concon_v1.co1_guard_battle_training import GuardBattleEnv, GAMMA, OPPONENTS, START_MODES
from concon_v1.evaluate_co1_guard import evaluate, print_summary

DEFAULT_EPISODES = 1000
CHECKPOINT_INTERVAL = 50
DEFAULT_EVAL_ROUNDS = 36


def epsilon_by_episode(episode, total):
    return max(0.05, 1.0 - 0.95 * episode / max(1, total * 0.7))


def curriculum_modes(episode, total):
    fraction = episode / max(1, total)
    if fraction <= 0.2:
        return ("hold",)
    if fraction <= 0.4:
        return ("hold", "transition")
    return START_MODES


def optimize(model, target, optimizer, replay, batch_size=64):
    if len(replay) < batch_size:
        return None
    batch = random.sample(replay, batch_size)
    obs, actions, rewards, next_obs, masks, dones, durations = zip(*batch)
    device = next(model.parameters()).device
    obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.int64, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    masks = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
    durations = torch.as_tensor(durations, dtype=torch.float32, device=device)
    selected = model(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        greedy = model(next_obs).masked_fill(~masks, -torch.inf).argmax(1)
        next_values = target(next_obs).gather(1, greedy[:, None]).squeeze(1)
        expected = rewards + GAMMA ** durations * next_values * (1 - dones)
    loss = nn.functional.smooth_l1_loss(selected, expected)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()
    return float(loss.item())


def make_checkpoint(model, scenario, episode, opponents, start_modes):
    return {"policy_type": "concon_guard_v1", "map_name": scenario.map_name,
            "scenario_signature": scenario.signature, "obs_dim": observation_dim(scenario),
            "n_actions": ACTION_DIM, "training_roster": list(GORIGONS.players),
            "attacker_perception": "production_iq", "episode": episode,
            "opponents": list(opponents), "start_modes": list(start_modes),
            "model_state_dict": {name: value.detach().cpu().clone()
                                 for name, value in model.state_dict().items()}}


def qualifies_as_best(candidate, current):
    return current is None or (candidate["mean_win_rate"] >= current["mean_win_rate"]
                               and candidate["min_team_win_rate"] >= current["min_team_win_rate"])


def train(episodes=DEFAULT_EPISODES, map_name="L", seed=0, save_dir=None, opponents=None,
          eval_rounds=DEFAULT_EVAL_ROUNDS, checkpoint_interval=CHECKPOINT_INTERVAL,
          start_modes=None, resume=None, device="cpu"):
    if episodes < 1 or eval_rounds < 1 or checkpoint_interval < 1:
        raise ValueError("episodes, eval rounds and checkpoint interval must be positive")
    scenario = get_scenario(map_name)
    opponents = tuple(opponents or OPPONENTS)
    if any(opponent not in OPPONENTS for opponent in opponents):
        raise ValueError("unknown guard opponent")
    if start_modes is not None and (not start_modes or any(mode not in START_MODES for mode in start_modes)):
        raise ValueError("unknown guard start mode")
    save_dir = Path(save_dir) if save_dir is not None else scenario.save_dir
    best_path = save_dir / scenario.checkpoint_filename("best")
    previous_best = None
    if best_path.exists():
        previous_best = best_path.read_bytes()
        validate_checkpoint(torch.load(io.BytesIO(previous_best), map_location="cpu", weights_only=False), scenario)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, target = GuardDQN(scenario).to(device), GuardDQN(scenario).to(device)
    first_episode = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, scenario)
        model.load_state_dict(checkpoint["model_state_dict"])
        first_episode = int(checkpoint.get("episode", 0))
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Float16 storage bounds memory; the model and optimization use float32.
    replay = deque(maxlen=5000)
    env = GuardBattleEnv(seed, opponents, model=model, map_name=scenario)
    recent = deque(maxlen=100)
    best_evaluation = None
    global_step = 0
    save_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(1, episodes + 1):
        episode = first_episode + offset
        modes = tuple(start_modes) if start_modes is not None else curriculum_modes(offset, episodes)
        initial = env.reset(start_mode=env.rng.choice(modes))
        epsilon = epsilon_by_episode(offset, episodes)
        reward_sum = 0.0
        while not env.done:
            transitions, rewards, _ = env.step(epsilon)
            replay.extend(transitions)
            reward_sum += sum(rewards)
            global_step += 1
            if global_step % 4 == 0:
                optimize(model, target, optimizer, replay)
            if global_step % 1000 == 0:
                target.load_state_dict(model.state_dict())
        result = env.result()
        recent.append(result)
        rate = sum(record["winner"] == "A" for record in recent) / len(recent)
        print(f"episode={episode} map={scenario.map_name} opponent={result['opponent']} "
              f"mode={result['start_mode']} alive_start={initial['attacker_alive']}:{initial['defender_alive']} "
              f"epsilon={epsilon:.3f} win100={rate:.3f} ticks={result['ticks']} "
              f"end={result['end_reason']} reward={reward_sum:.2f}", flush=True)
        with (save_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"episode": episode, "epsilon": epsilon, "reward": reward_sum,
                                     "initial": initial, **result}, ensure_ascii=False) + "\n")
        if offset % checkpoint_interval == 0 or offset == episodes:
            checkpoint = make_checkpoint(model, scenario, episode, opponents, START_MODES)
            checkpoint["evaluation"] = None
            torch.save(checkpoint, save_dir / scenario.checkpoint_filename("latest"))
            torch.save(checkpoint, save_dir / scenario.checkpoint_filename(f"episode_{episode}"))
            # Like route training, best selection starts at the exploration floor.
            if epsilon <= 0.05:
                if previous_best is not None and best_evaluation is None:
                    best_evaluation = evaluate(scenario, eval_rounds, seed, opponents,
                                               frozen_checkpoint=previous_best)
                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                evaluation = evaluate(scenario, eval_rounds, seed, opponents,
                                      frozen_checkpoint=buffer.getvalue())
                print_summary(evaluation)
                checkpoint["evaluation"] = evaluation
                torch.save(checkpoint, save_dir / scenario.checkpoint_filename("latest"))
                if qualifies_as_best(evaluation, best_evaluation):
                    best_evaluation = evaluation
                    torch.save(checkpoint, best_path)
                    print(f"Saved best: mean={evaluation['mean_win_rate']:.3f} "
                          f"min_team={evaluation['min_team_win_rate']:.3f}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-map", "--map", dest="map_name", choices=SCENARIOS, default="L")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--eval-rounds", type=int, default=DEFAULT_EVAL_ROUNDS)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--start-modes", nargs="+", choices=START_MODES,
                        help="override the default hold -> transition -> smoke curriculum")
    parser.add_argument("--resume", type=Path, help="warm start weights; replay and optimizer start fresh")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if min(args.episodes, args.eval_rounds, args.checkpoint_interval) < 1:
        parser.error("episode/evaluation/checkpoint counts must be positive")
    torch.set_num_threads(1)
    train(args.episodes, args.map_name, args.seed, args.save_dir, args.opponents,
          args.eval_rounds, args.checkpoint_interval, args.start_modes, args.resume, args.device)


if __name__ == "__main__":
    main()
