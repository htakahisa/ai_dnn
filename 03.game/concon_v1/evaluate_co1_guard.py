"""Evaluate frozen guard weights against the same five teams as attacker routes."""

import argparse
from collections import Counter
import hashlib
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

from concon_v1.co1_guard_common import GuardDQN, ACTION_DIM, observation_dim, GORIGONS
from concon_v1.co1_guard_scenarios import SCENARIOS, get_scenario, validate_checkpoint
from concon_v1.co1_guard_battle_training import GuardBattleEnv, OPPONENTS, START_MODES


def summarize(details):
    rounds = len(details)
    counts = Counter(record["end_reason"] for record in details)
    wins = sum(record["winner"] == "A" for record in details)
    return {"rounds": rounds, "attacker_wins": wins,
            "win_rate": wins / rounds if rounds else None,
            "defuse_rate": counts["defused"] / rounds if rounds else None,
            "end_reasons": dict(counts),
            "avg_ticks": sum(record["ticks"] for record in details) / rounds if rounds else None,
            "blocked_tap_decisions": sum(record["blocked_tap_decisions"] for record in details),
            "approach_decisions": sum(record["approach_decisions"] for record in details),
            "recon_on_tap": sum(record["recon_on_tap"] for record in details),
            "two_tick_fire_decisions": sum(record["two_tick_fire_decisions"] for record in details)}


def evaluate(map_name="L", rounds=36, seed=0, opponents=None, model_path=None,
             frozen_checkpoint=None, start_modes=START_MODES):
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not start_modes or any(mode not in START_MODES for mode in start_modes):
        raise ValueError("select known guard start modes")
    opponents = tuple(opponents or OPPONENTS)
    if not opponents or any(opponent not in OPPONENTS for opponent in opponents):
        raise ValueError("select known guard opponents")
    scenario = get_scenario(map_name)
    frozen_checkpoint = (bytes(frozen_checkpoint) if frozen_checkpoint is not None
                         else Path(model_path or scenario.model_path).read_bytes())
    checkpoint = torch.load(io.BytesIO(frozen_checkpoint), map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, scenario)
    if (checkpoint.get("obs_dim") != observation_dim(scenario)
            or checkpoint.get("n_actions") != ACTION_DIM
            or tuple(checkpoint.get("training_roster", ())) != GORIGONS.players):
        raise ValueError("guard checkpoint dimensions/roster do not match")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        model = GuardDQN(scenario)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        results = {}
        for opponent in opponents:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = GuardBattleEnv(seed, [opponent], model=model, map_name=scenario,
                                 start_modes=start_modes)
            details = []
            for trial in range(rounds):
                initial = env.reset(start_mode=start_modes[trial % len(start_modes)])
                while not env.done:
                    env.step(epsilon=0.0)
                details.append({**env.result(), "trial": trial + 1,
                                "initial_attacker_alive": initial["attacker_alive"],
                                "initial_defender_alive": initial["defender_alive"]})
            results[opponent] = {**summarize(details),
                                 "by_start_mode": {mode: summarize([record for record in details
                                                                    if record["start_mode"] == mode])
                                                   for mode in start_modes},
                                 "details": details}
        return {"map_name": scenario.map_name, "model_episode": checkpoint.get("episode"),
                "model_sha256": hashlib.sha256(frozen_checkpoint).hexdigest(),
                "epsilon": 0.0, "seed": seed, "rounds_per_opponent": rounds,
                "start_modes": list(start_modes), "opponents": results,
                "mean_win_rate": sum(result["win_rate"] for result in results.values()) / len(results),
                "min_team_win_rate": min(result["win_rate"] for result in results.values())}
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def print_summary(result):
    for opponent, counts in result["opponents"].items():
        print(f"map={result['map_name']} {opponent}: guard wins "
              f"{counts['attacker_wins']}/{counts['rounds']} ({counts['win_rate']:.1%}), "
              f"defused={counts['defuse_rate']:.1%} end={counts['end_reasons']}", flush=True)
        for mode, mode_counts in counts["by_start_mode"].items():
            if mode_counts["rounds"]:
                print(f"  {mode}: wins={mode_counts['attacker_wins']}/{mode_counts['rounds']} "
                      f"defused={mode_counts['defuse_rate']:.1%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-map", "--map", dest="map_name", choices=SCENARIOS, default="L")
    parser.add_argument("--rounds", type=int, default=36, help="postplant games per opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--start-modes", nargs="+", choices=START_MODES, default=list(START_MODES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    torch.set_num_threads(1)
    result = evaluate(args.map_name, args.rounds, args.seed, args.opponents,
                      args.model, start_modes=args.start_modes)
    print_summary(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
