"""Evaluate a frozen search checkpoint against each attacker team."""

import argparse
from collections import Counter
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

from concon_v1.co1_defender_scenario import get_scenario
from concon_v1.co1_defender_search_common import DefenderSearchBattleDQN, load_search_weights, SUPPORT_DISTANCE
from concon_v1.co1_defender_search_training import DefenderSearchEnv, START_MODES, OPPONENTS
from concon_v1.co1_defender_positioning import evaluate_positioning
from concon_v1.co1_defender_search_rewards import search_score

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def behavior_summary(records):
    def ratio(numerator, denominator):
        total = sum(record[denominator] for record in records)
        return sum(record[numerator] for record in records) / total if total else 0.
    normal_count = sum(record.get("normal_fire_decisions", record["fire_decisions"]) for record in records)
    moving = sum(record["moving_fire_decisions"] for record in records) / normal_count if normal_count else 0.
    normal_aligned = sum(record.get("aligned_normal_fire_decisions", record["aligned_fire_decisions"]) for record in records) / normal_count if normal_count else 0.
    bad_aim = 1 - ratio("aligned_fire_decisions", "fire_decisions") if any(r["fire_decisions"] for r in records) else 0.
    post_kill = ratio("stationary_aligned_post_kill", "post_kill_decisions")
    support = ratio("support_progress", "support_opportunities")
    memory_count = sum(record.get("memory_motion_decisions", 0) for record in records)
    memory_error = sum(record.get("memory_navigation_errors", 0) for record in records) / max(1, memory_count)
    return dict(moving_fire_rate=moving, stationary_aligned_fire_rate=1 - bad_aim,
                normal_stationary_aligned_fire_rate=normal_aligned,
                post_kill_hold_rate=post_kill, support_progress_rate=support,
                memory_return_facing_rate=ratio("aligned_memory_returns", "memory_return_decisions"),
                memory_motion_decisions=memory_count, memory_navigation_error_rate=memory_error,
                post_kill_decisions=sum(record["post_kill_decisions"] for record in records),
                flash_casts=sum(r["flash_casts"] for r in records), smoke_casts=sum(r["smoke_casts"] for r in records),
                enemy_flash_ticks=sum(r["enemy_flash_ticks"] for r in records),
                ally_flash_ticks=sum(r["ally_flash_ticks"] for r in records),
                # Shooting style is diagnostic only, including in tied-score selection.
                behavior_error=memory_error + (1 - post_kill if any(r["post_kill_decisions"] for r in records) else 0.)
                + (1 - support if any(r["support_opportunities"] for r in records) else 0.)
                + sum(r["smoke_casts"] for r in records) / max(1, len(records) * 5)
                + sum(r["distant_post_departures"] for r in records) / max(1, sum(r["search_ticks"] * 5 for r in records)))


def survival_summary(records):
    planted = [record for record in records if record["planted"]]
    count = len(planted)
    defender = sum(record["plant_defender_alive"] for record in planted) / count if count else None
    attacker = sum(record["plant_attacker_alive"] for record in planted) / count if count else None
    plant_scores = [search_score("planted", record["plant_defender_alive"], record["plant_attacker_alive"])
                    for record in planted]
    return dict(plant_rounds=count, mean_plant_defender_alive=defender, mean_plant_attacker_alive=attacker,
                mean_plant_search_score=sum(plant_scores) / count if count else None,
                mean_plant_advantage=defender - attacker if count else None,
                plant_advantage_rate=sum(record["plant_defender_alive"] > record["plant_attacker_alive"]
                                         for record in planted) / count if count else None,
                mean_search_score=sum(record["search_score"] for record in records) / len(records) if records else 0.)


def survival_log(counts):
    if not counts["plant_rounds"]:
        return "plant_alive_avg: defender=n/a attacker=n/a (plants=0) plant_search_score=n/a advantage_rate=n/a"
    return (f"{GREEN}plant_alive_avg: defender={counts['mean_plant_defender_alive']:.2f} "
            f"attacker={counts['mean_plant_attacker_alive']:.2f} (plants={counts['plant_rounds']}) "
            f"plant_search_score={counts['mean_plant_search_score']:.3f} "
            f"advantage_rate={counts['plant_advantage_rate']:.1%}{RESET}")


def evaluate(checkpoint_source=None, rounds=12, seed=0, opponents=None, start_modes=START_MODES,
             frozen_checkpoint=None, positioning_trials=5):
    if rounds < 1 or positioning_trials < 1 or not start_modes or any(mode not in START_MODES for mode in start_modes):
        raise ValueError("positive evaluation counts and known start modes required")
    opponents = tuple(OPPONENTS if opponents is None else opponents)
    if not opponents or any(opponent not in OPPONENTS for opponent in opponents):
        raise ValueError("select known defender search opponents")
    scenario = get_scenario()
    source = io.BytesIO(frozen_checkpoint) if frozen_checkpoint is not None else checkpoint_source or scenario.runtime_model_path
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        model = DefenderSearchBattleDQN(scenario)
        load_search_weights(model, checkpoint, scenario)
        model.foundation_only = checkpoint["policy_type"] == "concon_defender_search_positioning_v1"
        model.eval()
        positioning = evaluate_positioning(model, scenario, seed=seed + 7919, trials=positioning_trials)
        results, all_records = {}, []
        for opponent in opponents:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = DefenderSearchEnv(model, seed, [opponent], checkpoint.get("support_distance", SUPPORT_DISTANCE))
            records = []
            for trial in range(rounds):
                env.reset()
                while not env.done:
                    env.step(epsilon=0.)
                records.append(env.result())
            wins = sum(record["winner"] == "D" for record in records)
            results[opponent] = dict(rounds=rounds, defender_wins=wins, win_rate=wins / rounds,
                plant_rate=sum(record["planted"] for record in records) / rounds,
                end_reasons=dict(Counter(record["end_reason"] for record in records)),
                **behavior_summary(records), **survival_summary(records), details=records)
            all_records.extend(records)
        return dict(episode=checkpoint.get("episode", 0), seed=seed, epsilon=0.,
                    episode_start="normal_spawn_5v5_with_setup",
                    positioning=positioning, opponents=results, behavior=behavior_summary(all_records),
                    mean_win_rate=sum(result["win_rate"] for result in results.values()) / len(results),
                    min_team_win_rate=min(result["win_rate"] for result in results.values()),
                    mean_search_score=sum(result["mean_search_score"] for result in results.values()) / len(results),
                    min_team_search_score=min(result["mean_search_score"] for result in results.values()),
                    plant_survival=survival_summary(all_records))
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def print_summary(result):
    print(f"positioning passed={result['positioning']['passed']} "
          f"arrival={result['positioning']['arrival_rate']:.1%} "
          f"hold={result['positioning']['hold_rate']:.1%} facing={result['positioning']['facing_rate']:.1%}", flush=True)
    for opponent, counts in result["opponents"].items():
        print(f"{opponent}: defender wins {counts['defender_wins']}/{counts['rounds']} "
              f"({counts['win_rate']:.3f}) planted={counts['plant_rate']:.1%} "
              f"overall_search_score={counts['mean_search_score']:.3f} end={counts['end_reasons']}", flush=True)
        print(f"  {survival_log(counts)}", flush=True)
        print(f"  moving_fire={counts['moving_fire_rate']:.1%} post_kill_hold={counts['post_kill_hold_rate']:.1%} "
              f"normal_stop_aim={counts['normal_stationary_aligned_fire_rate']:.1%} "
              f"support_progress={counts['support_progress_rate']:.1%} "
              f"memory_move_error={counts['memory_navigation_error_rate']:.1%} "
              f"return_facing={counts['memory_return_facing_rate']:.1%} "
              f"flash={counts['flash_casts']} smoke={counts['smoke_casts']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--start-modes", nargs="+", choices=START_MODES, default=list(START_MODES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = evaluate(args.model, args.rounds, args.seed, args.opponents, args.start_modes)
    print_summary(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
