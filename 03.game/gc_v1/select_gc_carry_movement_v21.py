"""Select safe Carry movement rows against their baseline and package the winner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import evaluate_gc_curriculum_checkpoint as evaluator
import train_attacker_gc_real_curriculum as trainer


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def source_record(path, checkpoint):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "episode": checkpoint.get("episode"),
    }


def require_only_carry_movement_changed(baseline, candidate, label):
    movement_rows = set(trainer.MOVEMENT_ACTION_ROWS["carry"])
    for phase in trainer.PHASES:
        before = baseline[phase]["model_state_dict"]
        after = candidate[phase]["model_state_dict"]
        if before.keys() != after.keys():
            raise ValueError(f"{label}/{phase} state keys differ from baseline")
        for name in before:
            if phase == "carry" and name in (
                "advantage_head.2.weight",
                "advantage_head.2.bias",
            ):
                frozen = [
                    row
                    for row in range(before[name].shape[0])
                    if row not in movement_rows
                ]
                if not torch.equal(before[name][frozen], after[name][frozen]):
                    raise ValueError(
                        f"{label} changed frozen Carry action rows in {name}"
                    )
            elif name in ("feature.0.weight", "facing_feature.0.weight"):
                old, new = before[name], after[name]
                if old.shape[0] != new.shape[0] or old.shape[1] > new.shape[1]:
                    raise ValueError(f"{label} changed input shape {phase}/{name}")
                if not torch.equal(old, new[:, : old.shape[1]]):
                    raise ValueError(f"{label} changed frozen input {phase}/{name}")
            elif name in ("facing_head.weight", "facing_output.weight"):
                old, new = before[name], after[name]
                if old.shape[0] != new.shape[0] or old.shape[1] > new.shape[1]:
                    raise ValueError(f"{label} changed facing shape {phase}/{name}")
                if not torch.equal(old, new[:, : old.shape[1]]):
                    raise ValueError(f"{label} changed frozen facing {phase}/{name}")
            elif before[name].shape != after[name].shape:
                old, new = before[name], after[name]
                if old.ndim == new.ndim and all(a <= b for a, b in zip(old.shape, new.shape)):
                    if old.ndim == 1 and torch.equal(old, new[: old.shape[0]]):
                        continue
                    if old.ndim == 2 and torch.equal(old, new[: old.shape[0], : old.shape[1]]):
                        continue
                raise ValueError(f"{label} changed non-movement tensor {phase}/{name}")
            elif not torch.equal(before[name], after[name]):
                raise ValueError(
                    f"{label} changed non-movement tensor {phase}/{name}"
                )


def build_session(paths, checkpoints):
    policies = evaluator.build_policies(checkpoints)
    session = trainer.CurriculumSession(paths, policies, 0.99)
    for phase in trainer.PHASES:
        session.controllers[phase].positioning_version = checkpoints[phase].get(
            "positioning_version", 0
        )
        if phase in trainer.FACING_PHASES:
            session.controllers[phase].facing_head_enabled = (
                int(checkpoints[phase].get("facing_head_version", 0)) >= 1
            )
    return session


def evaluate_combo(paths, checkpoints, episodes, seeds):
    return trainer.evaluate_multi(
        build_session(paths, checkpoints), episodes, seeds, (100, 60, 40)
    )


def select_candidate(baseline_score, branches):
    """Prefer the unchanged baseline when a candidate only ties its score."""
    selected_score = baseline_score
    selected_label = "baseline"
    for label, branch in branches.items():
        if branch["score"] > selected_score:
            selected_score = branch["score"]
            selected_label = label
    return selected_score, selected_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--warm-dir", type=Path, required=True)
    parser.add_argument(
        "--reset-dir",
        type=Path,
        help="optional reset branch; omit after reset training has already been rejected",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-no-entry-rate", type=float, default=0.30)
    parser.add_argument("--max-timeout-rate", type=float, default=0.10)
    parser.add_argument("--holdout-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--holdout-episodes", type=int, default=50)
    args = parser.parse_args()
    if args.holdout_episodes < 1:
        parser.error("holdout-episodes must be positive")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("output-dir must be empty")

    baseline_paths = {
        phase: args.source_dir
        / f"dqn_attacker_{phase}_gc_best_phase_facing_ab.pt"
        for phase in trainer.PHASES
    }
    baseline = {phase: load_checkpoint(path) for phase, path in baseline_paths.items()}
    warm_history = json.loads(
        (args.warm_dir / "evaluation_history.json").read_text(encoding="utf-8")
    )
    if not warm_history:
        raise ValueError("warm branch has no baseline evaluation")
    baseline_metrics = warm_history[0]

    branch_directories = [("warm", args.warm_dir)]
    if args.reset_dir is not None:
        branch_directories.append(("reset", args.reset_dir))
    branches = {}
    for label, directory in branch_directories:
        paths = {
            phase: directory / f"dqn_attacker_{phase}_gc_best_by_eval.pt"
            for phase in trainer.PHASES
        }
        checkpoints = {phase: load_checkpoint(path) for phase, path in paths.items()}
        require_only_carry_movement_changed(baseline, checkpoints, label)
        metrics = checkpoints["carry"]["evaluation"]
        branches[label] = {
            "paths": paths,
            "checkpoints": checkpoints,
            "metrics": metrics,
            "score": trainer.carry_movement_selection_score(
                metrics, args.max_no_entry_rate, args.max_timeout_rate
            ),
        }

    baseline_score = trainer.carry_movement_selection_score(
        baseline_metrics, args.max_no_entry_rate, args.max_timeout_rate
    )
    selected_score, selected_label = select_candidate(baseline_score, branches)
    if selected_label == "baseline":
        selected_paths = baseline_paths
        selected_checkpoints = baseline
    else:
        selected_paths = branches[selected_label]["paths"]
        selected_checkpoints = branches[selected_label]["checkpoints"]

    holdout = evaluate_combo(
        selected_paths,
        selected_checkpoints,
        args.holdout_episodes,
        args.holdout_seeds,
    )
    fingerprint = trainer.runtime_data_fingerprint()
    report = {
        "revision": "carry_movement_rows_v21",
        "selection_metrics_source": "training evaluation seeds; holdout was not used for selection",
        "max_no_entry_rate": args.max_no_entry_rate,
        "max_timeout_rate": args.max_timeout_rate,
        "baseline": {
            "score": list(baseline_score),
            "evaluation": baseline_metrics,
            "sources": {
                phase: source_record(path, baseline[phase])
                for phase, path in baseline_paths.items()
            },
        },
        "branches": {
            label: {
                "score": list(branch["score"]),
                "evaluation": branch["metrics"],
                "sources": {
                    phase: source_record(branch["paths"][phase], branch["checkpoints"][phase])
                    for phase in trainer.PHASES
                },
            }
            for label, branch in branches.items()
        },
        "selected": selected_label,
        "selected_score": list(selected_score),
        "holdout_seeds": args.holdout_seeds,
        "holdout_episodes": args.holdout_episodes,
        "holdout": holdout,
        "runtime_data_fingerprint": fingerprint,
    }

    bundle = {
        "episode": selected_checkpoints["carry"].get("episode"),
        "selection_method": "carry_movement_rows_v21",
        "selected_branch": selected_label,
        "evaluation": holdout,
        "models": {},
        "runtime_data_fingerprint": fingerprint,
    }
    for phase in trainer.PHASES:
        payload = dict(selected_checkpoints[phase])
        payload.update(
            evaluation=holdout,
            selection_method="carry_movement_rows_v21",
            selected_branch=selected_label,
            runtime_data_fingerprint=fingerprint,
        )
        filename = f"dqn_attacker_{phase}_gc_best_carry_movement_ab.pt"
        torch.save(payload, args.output_dir / filename)
        bundle["models"][phase] = filename
    (args.output_dir / "best_carry_movement_ab_bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )
    (args.output_dir / "carry_movement_selection_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "holdout_carry_movement_ab.json").write_text(
        json.dumps(holdout, indent=2), encoding="utf-8"
    )
    print(
        "[CARRY MOVEMENT SELECTION] "
        + json.dumps(
            {
                "selected": selected_label,
                "score": list(selected_score),
                "baseline_score": list(baseline_score),
                "branch_scores": {
                    label: list(branch["score"])
                    for label, branch in branches.items()
                },
            }
        ),
        flush=True,
    )
    print("[CARRY MOVEMENT HOLDOUT] " + json.dumps(holdout), flush=True)


if __name__ == "__main__":
    main()
