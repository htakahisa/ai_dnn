"""Train one site/pattern guard model using the common postplant environment."""

import argparse
from collections import Counter, deque
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

from concon_v1.co1_guard_common import GuardDQN, ACTION_DIM, load_guard_weights, observation_dim, GORIGONS
from concon_v1.co1_guard_scenarios import SCENARIOS, get_scenario, validate_checkpoint
from concon_v1.co1_guard_battle_training import GuardBattleEnv, GAMMA, OPPONENTS, START_MODES
from concon_v1.evaluate_co1_guard import evaluate, print_summary
from concon_v1.co1_guard_rewards import REWARD_VERSION
from concon_v1.co1_guard_positioning import POSITIONING_VERSION, DEFAULT_POSITIONING_STEPS, prepare_positioning

DEFAULT_EPISODES = 1000
CHECKPOINT_INTERVAL = 50
DEFAULT_EVAL_ROUNDS = 36
TARGET_UPDATE_INTERVAL = 1000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_RATIO = 0.7
FORCE_SAVE = False  # Keep numbered debug models at every checkpoint interval.


def epsilon_by_episode(episode, total=DEFAULT_EPISODES):
    decay_episodes = max(1, int(total * EPSILON_DECAY_RATIO))
    fraction = min(max(float(episode) / decay_episodes, 0.0), 1.0)
    if fraction >= 1.0:
        return EPSILON_END
    return EPSILON_START + (EPSILON_END - EPSILON_START) * fraction


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
            "reward_version": REWARD_VERSION,
            "utility_policy": "learned_movement_wait_abilities_and_ultimates",
            "positioning_version": POSITIONING_VERSION if model.navigation else None,
            "positioning_training": getattr(model, "positioning_training", None),
            "battle_training_run": getattr(model, "battle_training_run", None),
            "battle_episodes_this_run": max(0, episode - getattr(model, "battle_training_run", {}).get("start_episode", episode)),
            "opponents": list(opponents), "start_modes": list(start_modes),
            "model_state_dict": {name: value.detach().cpu().clone()
                                 for name, value in model.state_dict().items()}}


def qualifies_as_best(candidate, current):
    if "positioning" in candidate and not candidate["positioning"]["passed"]:
        return False
    if current is None:
        return True
    if "positioning" in current and not current["positioning"]["passed"]:
        return True
    if (candidate["mean_win_rate"] < current["mean_win_rate"]
            or candidate["min_team_win_rate"] < current["min_team_win_rate"]):
        return False
    if (candidate["mean_win_rate"] > current["mean_win_rate"]
            or candidate["min_team_win_rate"] > current["min_team_win_rate"]):
        return True
    # On a win-rate plateau prefer stable holding and aim to oscillation.
    # Existing best weights are re-evaluated, so they also receive these metrics.
    return (candidate.get("behavior", {}).get("behavior_error", 0.0)
            <= current.get("behavior", {}).get("behavior_error", 0.0))


def print_training_summary(records, map_name, opponents, recent):
    """Report the completed checkpoint interval, grouped by opponent team."""
    rate = sum(record["winner"] == "A" for record in recent) / len(recent)
    print(f"episodes={records[0]['episode']}-{records[-1]['episode']} map={map_name} "
          f"rounds={len(records)} epsilon={records[-1]['epsilon']:.3f} "
          f"win100={rate:.3f}", flush=True)
    for opponent in opponents:
        team_records = [record for record in records if record["opponent"] == opponent]
        rounds = len(team_records)
        if not rounds:
            print(f"  {opponent}: rounds=0", flush=True)
            continue
        wins = sum(record["winner"] == "A" for record in team_records)
        avg_reward = sum(record["reward"] for record in team_records) / rounds
        avg_ticks = sum(record["ticks"] for record in team_records) / rounds
        modes = dict(Counter(record["start_mode"] for record in team_records))
        endings = dict(Counter(record["end_reason"] for record in team_records))
        print(f"  {opponent}: guard wins {wins}/{rounds} ({wins / rounds:.1%}) "
              f"avg_reward={avg_reward:.2f} avg_ticks={avg_ticks:.2f} "
              f"modes={modes} end={endings}", flush=True)


def train(episodes=DEFAULT_EPISODES, map_name="L", seed=0, save_dir=None, opponents=None,
          eval_rounds=DEFAULT_EVAL_ROUNDS, checkpoint_interval=CHECKPOINT_INTERVAL,
          start_modes=None, resume=None, device="cpu", positioning_only=False,
          positioning_steps=DEFAULT_POSITIONING_STEPS, retrain_positioning=False,
          force_save=FORCE_SAVE):
    if episodes < 1 or eval_rounds < 1 or checkpoint_interval < 1:
        raise ValueError("episodes, eval rounds and checkpoint interval must be positive")
    if positioning_only and resume is None:
        raise ValueError("--positioning-only requires --resume to preserve a trained combat policy")
    if positioning_steps < 0:
        raise ValueError("positioning steps cannot be negative")
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
    model, target = GuardDQN(scenario, navigation=True).to(device), GuardDQN(scenario, navigation=True).to(device)
    first_episode = 0
    reuse_positioning = False
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, scenario)
        if checkpoint.get("positioning_version") == POSITIONING_VERSION:
            load_guard_weights(model, checkpoint["model_state_dict"])
            reuse_positioning = True
        else:
            missing, unexpected = load_guard_weights(model, checkpoint["model_state_dict"], strict=False)
            if set(missing) != {"navigation_values.weight", "navigation_other_posts", "navigation_yield_values.weight"} or unexpected:
                raise ValueError("incompatible guard warm-start weights")
        first_episode = int(checkpoint.get("episode", 0))
    positioning, positioning_training = prepare_positioning(
        model, scenario, steps=positioning_steps, reuse=reuse_positioning,
        force=retrain_positioning, seed=seed + 7919)
    model.positioning_training = positioning_training
    model.battle_training_run = {"start_episode": first_episode,
                                "requested_additional_episodes": 0 if positioning_only else episodes,
                                "seed": seed, "eval_rounds_per_opponent": eval_rounds}
    print(f"Positioning: {positioning_training} validation={positioning}", flush=True)
    print(f"Battle training: additional_episodes={0 if positioning_only else episodes} "
          f"start_episode={first_episode} final_episode={first_episode + (0 if positioning_only else episodes)} "
          f"evaluation_rounds_per_opponent={eval_rounds}", flush=True)
    # Preserve the learned basic skill while the perception network learns
    # combat and defuse responses. Quiet actions remain model predictions.
    model.navigation_values.weight.requires_grad_(False)
    model.navigation_yield_values.weight.requires_grad_(False)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Float16 storage bounds memory; the model and optimization use float32.
    replay = deque(maxlen=5000)
    env = GuardBattleEnv(seed, opponents, model=model, map_name=scenario)
    recent = deque(maxlen=100)
    interval_records = []
    best_evaluation = None
    global_step = 0
    save_dir.mkdir(parents=True, exist_ok=True)
    foundation = make_checkpoint(model, scenario, first_episode, opponents, START_MODES)
    foundation["positioning"] = positioning
    torch.save(foundation, save_dir / scenario.checkpoint_filename("positioning"))
    if positioning_only:
        buffer = io.BytesIO()
        torch.save(foundation, buffer)
        evaluation = evaluate(scenario, eval_rounds, seed, opponents, frozen_checkpoint=buffer.getvalue())
        foundation["evaluation"] = evaluation
        print_summary(evaluation)
        torch.save(foundation, save_dir / scenario.checkpoint_filename("latest"))
        if previous_best is not None:
            best_evaluation = evaluate(scenario, eval_rounds, seed, opponents, frozen_checkpoint=previous_best)
        if qualifies_as_best(evaluation, best_evaluation):
            if previous_best is not None:
                backup = save_dir / scenario.checkpoint_filename("before_positioning_v1")
                if not backup.exists():
                    backup.write_bytes(previous_best)
            torch.save(foundation, best_path)
            print(f"Saved positioning-validated best: {best_path}", flush=True)
        return model
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
            if global_step % TARGET_UPDATE_INTERVAL == 0:
                target.load_state_dict(model.state_dict())
        result = env.result()
        recent.append(result)
        record = {"episode": episode, "epsilon": epsilon, "reward": reward_sum,
                  "initial": initial, **result}
        interval_records.append(record)
        with (save_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if offset % checkpoint_interval == 0 or offset == episodes:
            print_training_summary(interval_records, scenario.map_name, opponents, recent)
            interval_records.clear()
            checkpoint = make_checkpoint(model, scenario, episode, opponents, START_MODES)
            checkpoint["epsilon"] = epsilon
            checkpoint["epsilon_end"] = EPSILON_END
            checkpoint["evaluation"] = None
            torch.save(checkpoint, save_dir / scenario.checkpoint_filename("latest"))
            if force_save:
                debug_path = save_dir / scenario.checkpoint_filename(f"episode_{episode}")
                torch.save(checkpoint, debug_path)
                print(f"Force-saved debug model at episode {episode} "
                      f"epsilon={epsilon:.3f}: {debug_path}", flush=True)
            # Like route training, best selection starts at the exploration floor.
            if epsilon <= EPSILON_END:
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
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="additional battle episodes in this run (also when resuming)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--eval-rounds", type=int, default=DEFAULT_EVAL_ROUNDS)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--force-save", action=argparse.BooleanOptionalAction, default=FORCE_SAVE,
                        help="keep numbered debug models at every checkpoint interval regardless of epsilon")
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--start-modes", nargs="+", choices=START_MODES,
                        help="override the default hold -> transition -> smoke curriculum")
    parser.add_argument("--resume", type=Path, help="warm start weights; replay and optimizer start fresh")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--positioning-only", action="store_true",
                        help="learn/validate positioning and evaluate/save it without battle optimization")
    parser.add_argument("--positioning-steps", type=int, default=DEFAULT_POSITIONING_STEPS,
                        help="supervised updates if positioning needs training; 0 requires a validated resume")
    parser.add_argument("--retrain-positioning", action="store_true",
                        help="train positioning again even if resumed positioning passes validation")
    args = parser.parse_args()
    if min(args.episodes, args.eval_rounds, args.checkpoint_interval) < 1:
        parser.error("episode/evaluation/checkpoint counts must be positive")
    if args.positioning_steps < 0:
        parser.error("positioning steps cannot be negative")
    torch.set_num_threads(1)
    train(args.episodes, args.map_name, args.seed, args.save_dir, args.opponents,
          args.eval_rounds, args.checkpoint_interval, args.start_modes, args.resume, args.device,
          args.positioning_only, args.positioning_steps, args.retrain_positioning,
          force_save=args.force_save)


if __name__ == "__main__":
    main()
