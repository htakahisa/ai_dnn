"""Manually train defender search, or explicitly rebuild its basic positioning."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from concon_v1.co1_defender_common import DefenderSearchDQN, observation_dim, ACTION_DIM, GORIGONS
from concon_v1.co1_defender_scenario import get_scenario, validate_checkpoint
from concon_v1.co1_defender_positioning import learn_positioning, evaluate_positioning, POSITIONING_VERSION

# 学習設定はここで変更する。コマンドライン引数がある設定は引数を優先する。
DEFAULT_EPISODES = 1000
CHECKPOINT_INTERVAL = 50
DEFAULT_EVAL_ROUNDS = 12  # 対戦相手ごとの評価回数。
DEFAULT_POSITIONING_STEPS = 250
DEFAULT_EVAL_TRIALS = 20
POSITIONING_CHECK_TRIALS = 5
DEFAULT_SUPPORT_DISTANCE = 7
EPSILON_START = 1.
EPSILON_END = .05
EPSILON_DECAY_RATIO = .7
TARGET_UPDATE_INTERVAL = 1000
REPLAY_SIZE = 5000
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
POSITIONING_LEARNING_RATE = .1
OPTIMIZE_INTERVAL = 4
GRAD_CLIP_NORM = 10
COMBAT_ACTION_MARGIN = .25
COMBAT_LOSS_WEIGHT = .5


def make_checkpoint(model, scenario, result, steps):
    return dict(policy_type="concon_defender_search_positioning_v1",
                scenario_signature=scenario.signature, positioning_version=POSITIONING_VERSION,
                obs_dim=observation_dim(scenario), n_actions=ACTION_DIM,
                training_roster=list(GORIGONS.players), positioning=result,
                positioning_steps=steps, defender_perception="production_iq_live",
                training_method="bfs_supervised_positioning_like_guard",
                phase_scope="setup_and_live_positioning_only",
                setup_positions=scenario.setup_positions,
                model_state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()})


def train(steps=DEFAULT_POSITIONING_STEPS, seed=0, save_dir=None, resume=None, trials=DEFAULT_EVAL_TRIALS, device="cpu"):
    if steps < 0 or trials < 1 or (steps == 0 and resume is None):
        raise ValueError("positive updates required for a new model; evaluation trials must be positive")
    torch.manual_seed(seed)
    scenario = get_scenario()
    model = DefenderSearchDQN(scenario).to(device)
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, scenario)
        model.load_state_dict(checkpoint["model_state_dict"])
    if steps:
        learn_positioning(model, scenario, steps, learning_rate=POSITIONING_LEARNING_RATE)
    result = evaluate_positioning(model, scenario, seed=seed, trials=trials)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise RuntimeError("defender basic positioning failed validation; no checkpoint published")
    directory = Path(save_dir) if save_dir is not None else scenario.save_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / scenario.model_path.name
    torch.save(make_checkpoint(model, scenario, result, steps), path)
    print(f"Saved defender basic model: {path.resolve()}", flush=True)
    return model


def main(argv=None):
    from concon_v1.co1_defender_search_battle_training import train_battle
    from concon_v1.co1_defender_search_training import OPPONENTS, START_MODES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("battle", "positioning"), default="battle",
                        help="battle adds search learning to the saved foundation; positioning rebuilds the basic skill")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="additional search battle episodes, including when resuming")
    parser.add_argument("--eval-rounds", type=int, default=DEFAULT_EVAL_ROUNDS,
                        help="full rounds per attacker opponent in checkpoint evaluation")
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--support-distance", type=int, default=DEFAULT_SUPPORT_DISTANCE)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--start-modes", nargs="+", choices=START_MODES, default=["round"],
                        help="normal 5v5 round from spawn with defender setup")
    parser.add_argument("--force-save", action="store_true", help="keep numbered checkpoints at each interval")
    parser.add_argument("--positioning-steps", type=int, default=DEFAULT_POSITIONING_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-trials", type=int, default=DEFAULT_EVAL_TRIALS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.mode == "positioning":
        train(args.positioning_steps, args.seed, args.save_dir, args.resume, args.eval_trials, args.device)
    else:
        train_battle(episodes=args.episodes, seed=args.seed, save_dir=args.save_dir, resume=args.resume,
                     opponents=args.opponents, eval_rounds=args.eval_rounds,
                     checkpoint_interval=args.checkpoint_interval, support_distance=args.support_distance,
                     device=args.device, start_modes=args.start_modes, force_save=args.force_save)


if __name__ == "__main__":
    main()
