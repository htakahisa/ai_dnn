"""Manually train the left/right basic movement and defuse models."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from concon_v1.co1_retake_config import DEFAULT_ABILITY_DISTANCES, add_ability_arguments, ability_distances_from_args
from concon_v1.co1_retake_common import RetakeDQN
from concon_v1.co1_retake_scenarios import get_scenario, normalize_site_ability_distances, make_checkpoint
from concon_v1.co1_retake_foundation import learn_foundation, evaluate_foundation
from concon_v1.co1_retake_start_positions import validate_starts

DEFAULT_FOUNDATION_STEPS = 250
DEFAULT_FOUNDATION_TRIALS = 20
FOUNDATION_LEARNING_RATE = .1


def train_foundation(steps=DEFAULT_FOUNDATION_STEPS, trials=DEFAULT_FOUNDATION_TRIALS,
                     seed=0, save_dir=None, ability_distance=DEFAULT_ABILITY_DISTANCES, device="cpu"):
    if steps < 1 or trials < 5:
        raise ValueError("positive foundation steps and at least five evaluation trials are required")
    distances = normalize_site_ability_distances(ability_distance)
    torch.manual_seed(seed)
    starts = validate_starts()
    print(f"Retake foundation: updates={steps} evaluation_trials_per_site={trials}", flush=True)
    for slot, cells in starts.items():
        print(f"  search post {slot}: start_cells={cells}", flush=True)
    models, checkpoints = {}, {}
    for site in ("L", "R"):
        scenario = get_scenario(site)
        model = RetakeDQN(scenario, foundation=True).to(device)
        result = learn_foundation(model, scenario, steps, FOUNDATION_LEARNING_RATE, seed)
        evaluation = evaluate_foundation(model, scenario, trials, seed + 7919, distances[site])
        print(f"  {site}: training_states={result['training_states']} loss={result['loss']:.6f} "
              f"arrival={evaluation['arrival_rate']:.1%} defuse={evaluation['defuse_rate']:.1%} "
              f"passed={evaluation['passed']}", flush=True)
        if not evaluation["passed"]:
            failures = [row for row in evaluation["details"] if not row["defused"]]
            raise RuntimeError(f"{site} foundation validation failed; no foundation checkpoints saved: {failures}")
        model.foundation_evaluation = evaluation
        checkpoint = make_checkpoint(model, site, 0, distances, "not_used_in_foundation", [], 0)
        checkpoint.update(phase_scope="quiet_movement_and_defuse_foundation", foundation_training=result,
                          foundation_start_cells=starts, training_method="bfs_supervised_learned_values")
        models[site], checkpoints[site] = model, checkpoint
    for site, checkpoint in checkpoints.items():
        scenario = get_scenario(site)
        directory = Path(save_dir) if save_dir is not None else scenario.save_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / scenario.model_path("foundation").name
        torch.save(checkpoint, path)
        print(f"Saved {site} foundation: {path.resolve()}", flush=True)
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", "--foundation-steps", type=int, default=DEFAULT_FOUNDATION_STEPS)
    parser.add_argument("--eval-trials", "--foundation-trials", type=int, default=DEFAULT_FOUNDATION_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    add_ability_arguments(parser)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    return train_foundation(args.steps, args.eval_trials, args.seed, args.save_dir,
                            ability_distances_from_args(args), args.device)


if __name__ == "__main__":
    main()
