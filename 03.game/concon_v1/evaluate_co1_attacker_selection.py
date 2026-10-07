"""Evaluate A1-A4 against each opponent and publish runtime route selection."""

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from uuid import uuid4
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from concon_v1.co1_attacker_scenarios import SCENARIOS, get_scenario, validate_checkpoint_scenario
from concon_v1.co1_attacker_selection import METRIC, SELECTION_PATH
from concon_v1.co1_battle_training import OPPONENTS, _run_from_project_root


@_run_from_project_root
def _load_evaluator():
    # Legacy opponents resolve model/data paths during module import too.
    with contextlib.redirect_stdout(io.StringIO()):
        from concon_v1.evaluate_co1_attacker import DEFAULT_ROUNDS, evaluate
    return DEFAULT_ROUNDS, evaluate


DEFAULT_ROUNDS, evaluate = _load_evaluator()


def _evaluate_pair(name, opponent, rounds, seed, data):
    torch.set_num_threads(1)
    started = time.perf_counter()

    def progress(trial, limit):
        if trial % 5 == 0 or trial == limit:
            print(f"  {name}/{opponent}: {trial}/{limit} elapsed={time.perf_counter() - started:.0f}s", flush=True)

    return evaluate(opponent, rounds, seed, frozen_checkpoint=data, map_name=name, on_trial=progress)


def evaluate_selection(rounds=DEFAULT_ROUNDS, seed=0, threshold=.8, output=SELECTION_PATH, workers=1):
    if rounds < 1 or workers < 1 or not 0 <= threshold <= 1:
        raise ValueError("rounds/workers must be positive and threshold must be between 0 and 1")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Check the destination before running hundreds of matches.
    probe = output.parent / f".{output.name}.{uuid4().hex}.probe"
    with probe.open("x", encoding="utf-8") as handle:
        handle.write("write check")
    probe.unlink()
    torch.set_num_threads(1)
    frozen = {}
    # Freeze and validate every model before spending time on matches.
    for name in SCENARIOS:
        scenario = get_scenario(name)
        path = scenario.model_path
        if not path.is_file():
            path = scenario.save_dir / scenario.checkpoint_filename("latest")
        data = path.read_bytes()
        validate_checkpoint_scenario(torch.load(io.BytesIO(data), map_location="cpu", weights_only=False), scenario)
        frozen[name] = data
    results = []
    started = time.perf_counter()
    total = len(SCENARIOS) * len(OPPONENTS)
    pairs = [(name, opponent) for name in SCENARIOS for opponent in OPPONENTS]

    def record(result):
        results.append(result)
        print(f"[{len(results)}/{total}] {result['map_name']}/{result['opponent']}: "
              f"success={result['attack_successes']}/{rounds} ({result['attack_success_rate']:.1%}) "
              f"plants={result['plants']} preplant_eliminations={result['preplant_defender_eliminations']} "
              f"elapsed={time.perf_counter() - started:.0f}s", flush=True)

    if workers == 1:
        for name, opponent in pairs:
            record(_evaluate_pair(name, opponent, rounds, seed, frozen[name]))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, total)) as pool:
            futures = [pool.submit(_evaluate_pair, name, opponent, rounds, seed, frozen[name])
                       for name, opponent in pairs]
            for future in as_completed(futures):
                record(future.result())
    order = {pair: index for index, pair in enumerate(pairs)}
    results.sort(key=lambda row: order[row["map_name"], row["opponent"]])
    report = dict(version=1, metric=METRIC, threshold=threshold, seed=seed,
                  created_at=datetime.now(timezone.utc).isoformat(), results=results)
    staging = output.with_suffix(output.suffix + ".tmp")
    staging.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    staging.replace(output)
    for opponent in OPPONENTS:
        rows = [row for row in results if row["opponent"] == opponent]
        eligible = [row["map_name"] for row in rows if row["attack_success_rate"] >= threshold]
        best = max(row["attack_success_rate"] for row in rows)
        candidates = eligible or [row["map_name"] for row in rows if row["attack_success_rate"] == best]
        print(f"{opponent}: candidates={','.join(candidates)} "
              f"selection={'threshold' if eligible else 'best_available'}", flush=True)
    print(f"Saved runtime evaluation: {output.resolve()}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="rounds per route per opponent (default: 36)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--workers", type=int, default=6, help="independent evaluation processes (default: 6)")
    parser.add_argument("--output", type=Path, default=SELECTION_PATH)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.workers < 1 or not 0 <= args.threshold <= 1:
        parser.error("rounds/workers must be positive and threshold must be between 0 and 1")
    evaluate_selection(args.rounds, args.seed, args.threshold, args.output, args.workers)


if __name__ == "__main__":
    main()
