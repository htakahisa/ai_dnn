"""Select Carry/Escort facing heads with phase-isolated A/B evaluations."""
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


def checkpoint_path(models_dir, phase, kind):
    return models_dir / f"dqn_attacker_{phase}_gc_{kind}.pt"


def movement_state(checkpoint):
    return {
        name: value
        for name, value in checkpoint["model_state_dict"].items()
        if not trainer.is_facing_parameter(name)
    }


def require_same_movement(baseline, candidate, phase, kind):
    before, after = movement_state(baseline), movement_state(candidate)
    if before.keys() != after.keys() or any(
        not torch.equal(before[name], after[name]) for name in before
    ):
        raise ValueError(
            f"{phase}/{kind} changes movement tensors; refusing facing-only A/B"
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
    session = build_session(paths, checkpoints)
    return trainer.evaluate_multi(session, episodes, seeds, (100, 60, 40))


def source_record(path, checkpoint):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "episode": checkpoint.get("episode"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-kinds", nargs="+", default=("best_by_eval", "latest")
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--holdout-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--holdout-episodes", type=int, default=50)
    parser.add_argument("--regression-tolerance", type=float, default=0.05)
    parser.add_argument("--allow-data-revision-mismatch", action="store_true")
    parser.add_argument(
        "--report", type=Path,
        help="incremental report; defaults inside models-dir",
    )
    args = parser.parse_args()
    if args.eval_episodes < 1 or args.holdout_episodes < 1:
        parser.error("episode counts must be positive")
    if not 0 <= args.regression_tolerance <= 1:
        parser.error("regression tolerance must be in [0, 1]")
    torch.set_num_threads(1)
    args.models_dir = args.models_dir.resolve()
    report_path = args.report or args.models_dir / "phase_facing_ab_report.json"

    baseline_paths = {
        phase: checkpoint_path(
            args.models_dir,
            phase,
            "best_facing" if phase in trainer.FACING_PHASES else "best_by_eval",
        )
        for phase in trainer.PHASES
    }
    missing = [str(path) for path in baseline_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"baseline checkpoints are missing: {missing}")
    baseline_checkpoints = {
        phase: load_checkpoint(path) for phase, path in baseline_paths.items()
    }
    training_data_fingerprint, evaluation_data_fingerprint = (
        evaluator.validate_runtime_data(
            baseline_checkpoints, args.allow_data_revision_mismatch
        )
    )

    report = {
        "revision": "phase_isolated_facing_ab_v1",
        "eval_seeds": args.eval_seeds,
        "eval_episodes": args.eval_episodes,
        "holdout_seeds": args.holdout_seeds,
        "holdout_episodes": args.holdout_episodes,
        "regression_tolerance": args.regression_tolerance,
        "training_runtime_data_fingerprint": training_data_fingerprint,
        "evaluation_runtime_data_fingerprint": evaluation_data_fingerprint,
        "cross_revision_evaluation": (
            training_data_fingerprint is not None
            and training_data_fingerprint != evaluation_data_fingerprint
        ),
        "baseline_sources": {
            phase: source_record(baseline_paths[phase], baseline_checkpoints[phase])
            for phase in trainer.PHASES
        },
        "evaluations": {},
    }

    def persist():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    baseline_metrics = evaluate_combo(
        baseline_paths,
        baseline_checkpoints,
        args.eval_episodes,
        args.eval_seeds,
    )
    report["baseline_evaluation"] = baseline_metrics
    persist()
    print("[AB BASELINE] " + json.dumps(baseline_metrics), flush=True)

    selected = {
        phase: {
            "kind": "best_facing",
            "path": baseline_paths[phase],
            "checkpoint": baseline_checkpoints[phase],
            "metrics": baseline_metrics,
            "score": trainer.phase_facing_ab_score(
                phase, baseline_metrics, baseline_metrics, args.regression_tolerance
            ),
        }
        for phase in trainer.FACING_PHASES
    }

    for phase in trainer.FACING_PHASES:
        report["evaluations"][phase] = []
        for kind in args.candidate_kinds:
            path = checkpoint_path(args.models_dir, phase, kind)
            if not path.is_file():
                raise FileNotFoundError(path)
            candidate = load_checkpoint(path)
            require_same_movement(
                baseline_checkpoints[phase], candidate, phase, kind
            )
            paths = dict(baseline_paths)
            checkpoints = dict(baseline_checkpoints)
            paths[phase] = path
            checkpoints[phase] = candidate
            evaluator.validate_runtime_data(
                checkpoints, args.allow_data_revision_mismatch
            )
            metrics = evaluate_combo(
                paths, checkpoints, args.eval_episodes, args.eval_seeds
            )
            score = trainer.phase_facing_ab_score(
                phase, metrics, baseline_metrics, args.regression_tolerance
            )
            entry = {
                "kind": kind,
                "source": source_record(path, candidate),
                "score": list(score),
                "safety_passed": bool(score[0]),
                "evaluation": metrics,
            }
            report["evaluations"][phase].append(entry)
            persist()
            print(
                f"[AB {phase.upper()} {kind}] " + json.dumps(entry), flush=True
            )
            if score > selected[phase]["score"]:
                selected[phase] = {
                    "kind": kind,
                    "path": path,
                    "checkpoint": candidate,
                    "metrics": metrics,
                    "score": score,
                }

    selected_paths = dict(baseline_paths)
    selected_checkpoints = dict(baseline_checkpoints)
    for phase in trainer.FACING_PHASES:
        selected_paths[phase] = selected[phase]["path"]
        selected_checkpoints[phase] = selected[phase]["checkpoint"]
    composite = evaluate_combo(
        selected_paths,
        selected_checkpoints,
        args.eval_episodes,
        args.eval_seeds,
    )
    selected_episodes = {
        phase: selected_checkpoints[phase].get("episode")
        for phase in trainer.FACING_PHASES
    }
    composite_episode = max(int(value or 0) for value in selected_episodes.values())
    report["selection"] = {
        phase: {
            "kind": selected[phase]["kind"],
            "episode": selected_episodes[phase],
            "score": list(selected[phase]["score"]),
        }
        for phase in trainer.FACING_PHASES
    }
    report["composite_evaluation"] = composite
    report["composite_episode"] = composite_episode
    persist()

    bundle = {
        "episode": composite_episode,
        "selected_episodes_by_phase": selected_episodes,
        "selection_method": "phase_isolated_facing_ab_v1",
        "evaluation": composite,
        "models": {},
        "runtime_data_fingerprint": trainer.runtime_data_fingerprint(),
    }
    for phase in trainer.PHASES:
        payload = dict(selected_checkpoints[phase])
        payload["episode"] = composite_episode
        payload["evaluation"] = composite
        payload["selection_method"] = "phase_isolated_facing_ab_v1"
        payload["selected_facing_episode"] = selected_episodes.get(phase)
        payload["phase_facing_ab_report"] = str(report_path)
        filename = f"dqn_attacker_{phase}_gc_best_phase_facing_ab.pt"
        torch.save(payload, args.models_dir / filename)
        bundle["models"][phase] = filename
    (args.models_dir / "best_phase_facing_ab_bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )

    holdout = evaluate_combo(
        selected_paths,
        selected_checkpoints,
        args.holdout_episodes,
        args.holdout_seeds,
    )
    holdout["selected_episodes_by_phase"] = selected_episodes
    report["holdout"] = holdout
    persist()
    (args.models_dir / "holdout_phase_facing_ab.json").write_text(
        json.dumps(holdout, indent=2), encoding="utf-8"
    )
    print("[AB SELECTION] " + json.dumps(report["selection"]), flush=True)
    print("[AB COMPOSITE] " + json.dumps(composite), flush=True)
    print("[AB HOLDOUT] " + json.dumps(holdout), flush=True)


if __name__ == "__main__":
    main()
