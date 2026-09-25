"""Package a safe joint Carry/Escort movement checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import select_gc_carry_movement_v21 as common
import train_attacker_gc_real_curriculum as trainer


def require_only_escort_movement_changed(baseline, candidate):
    rows = set(trainer.MOVEMENT_ACTION_ROWS["escort"])
    for phase in trainer.PHASES:
        before = baseline[phase]["model_state_dict"]
        after = candidate[phase]["model_state_dict"]
        if before.keys() != after.keys():
            raise ValueError(f"{phase} state keys differ from baseline")
        for name in before:
            if phase == "escort" and name in (
                "advantage_head.2.weight",
                "advantage_head.2.bias",
            ):
                frozen_rows = [
                    row for row in range(before[name].shape[0]) if row not in rows
                ]
                if not torch.equal(before[name][frozen_rows], after[name][frozen_rows]):
                    raise ValueError(f"Escort frozen action rows changed in {name}")
            elif phase == "escort" and name in (
                "feature.0.weight", "facing_feature.0.weight"
            ):
                old = before[name]
                new = after[name]
                old_width = old.shape[1]
                new_start = trainer.escort_runtime.FACING_HEAD_OBS_DIM
                frozen_width = min(old_width, new_start)
                if (
                    old.shape[0] != new.shape[0]
                    or old_width > new.shape[1]
                    or new.shape[1] > trainer.escort_runtime.FAKE_WAIT_SUPPORT_OBS_DIM
                    or not torch.equal(old[:, :frozen_width], new[:, :frozen_width])
                ):
                    raise ValueError(f"Escort pretrained input weights changed in {name}")
                if old_width < new_start and not torch.equal(
                    new[:, old_width:new_start],
                    torch.zeros_like(new[:, old_width:new_start]),
                ):
                    raise ValueError(f"Escort unrelated input weights changed in {name}")
                if name == "facing_feature.0.weight":
                    if not torch.equal(old, new[:, :old_width]) or not torch.equal(
                        new[:, old_width:], torch.zeros_like(new[:, old_width:])
                    ):
                        raise ValueError("Escort facing input weights changed")
            elif not torch.equal(before[name], after[name]):
                raise ValueError(f"Non-Escort-movement tensor changed: {phase}/{name}")


def require_only_carry_escort_movement_changed(
    baseline, candidate, allow_facing_changes=False,
):
    """Allow movement/orb updates and optional independent facing updates."""
    movement_rows = {
        phase: set(trainer.MOVEMENT_ACTION_ROWS[phase])
        for phase in ("carry", "escort")
    }
    for phase in trainer.PHASES:
        before = baseline[phase]["model_state_dict"]
        after = candidate[phase]["model_state_dict"]
        if before.keys() != after.keys():
            raise ValueError(f"{phase} state keys differ from baseline")
        for name in before:
            old, new = before[name], after[name]
            if (
                allow_facing_changes
                and phase in movement_rows
                and name.startswith(("facing_feature.", "facing_head.", "facing_output."))
            ):
                if old.ndim != new.ndim or any(
                    before_dim > after_dim
                    for before_dim, after_dim in zip(old.shape, new.shape)
                ):
                    raise ValueError(f"{phase} facing tensor shape changed unexpectedly: {name}")
                continue
            if name in ("feature.0.weight", "facing_feature.0.weight"):
                if old.shape[0] != new.shape[0] or old.shape[1] > new.shape[1]:
                    raise ValueError(f"{phase} input shape changed unexpectedly: {name}")
                if phase in movement_rows and name == "feature.0.weight":
                    # The orb approach feature is appended after the facing
                    # input; existing orb columns also train with movement.
                    mutable_start = (
                        trainer.runtime.FACING_HEAD_OBS_DIM if phase == "carry"
                        else trainer.escort_runtime.FACING_HEAD_OBS_DIM
                    )
                    frozen = min(old.shape[1], mutable_start)
                    if not torch.equal(old[:, :frozen], new[:, :frozen]):
                        raise ValueError(f"Frozen tensor changed: {phase}/{name}")
                    continue
                if not torch.equal(old, new[:, : old.shape[1]]):
                    raise ValueError(f"Frozen tensor changed: {phase}/{name}")
                continue
            if name in ("facing_head.weight", "facing_output.weight"):
                if old.shape[0] != new.shape[0] or old.shape[1] > new.shape[1]:
                    raise ValueError(f"{phase} facing shape changed unexpectedly: {name}")
                if not torch.equal(old, new[:, : old.shape[1]]):
                    raise ValueError(f"Frozen tensor changed: {phase}/{name}")
                continue
            if name in ("advantage_head.2.weight", "advantage_head.2.bias"):
                frozen_rows = [row for row in range(old.shape[0])
                               if phase not in movement_rows or row not in movement_rows[phase]]
                if not torch.equal(old[frozen_rows], new[frozen_rows]):
                    raise ValueError(f"{phase} frozen action rows changed in {name}")
                continue
            if old.shape != new.shape:
                if old.ndim == new.ndim and all(a <= b for a, b in zip(old.shape, new.shape)):
                    if old.ndim == 1 and torch.equal(old, new[: old.shape[0]]):
                        continue
                    if old.ndim == 2 and torch.equal(old, new[: old.shape[0], : old.shape[1]]):
                        continue
            elif torch.equal(old, new):
                continue
            else:
                raise ValueError(f"Frozen tensor changed: {phase}/{name}")
            raise ValueError(f"Frozen tensor changed: {phase}/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--source-kind", default="best_carry_movement_ab")
    for phase in trainer.PHASES:
        parser.add_argument("--source-" + phase, type=Path)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--allow-data-revision-mismatch", action="store_true")
    parser.add_argument(
        "--allow-facing-changes",
        action="store_true",
        help="accept Carry/Escort independent-facing tensor updates",
    )
    parser.add_argument("--max-no-entry-rate", type=float, default=0.30)
    parser.add_argument("--max-timeout-rate", type=float, default=0.10)
    parser.add_argument("--max-quiet-stall-increase", type=float, default=0.05)
    parser.add_argument("--max-carrier-death-increase", type=float, default=0.05)
    parser.add_argument("--max-plant-regression", type=float, default=0.05)
    parser.add_argument("--holdout-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--holdout-episodes", type=int, default=50)
    args = parser.parse_args()
    if args.holdout_episodes < 1:
        parser.error("holdout-episodes must be positive")
    torch.set_num_threads(1)
    source_paths = {}
    for phase in trainer.PHASES:
        explicit = getattr(args, "source_" + phase)
        if explicit is None and args.source_dir is None:
            parser.error(f"--source-{phase} or --source-dir is required")
        source_paths[phase] = (
            explicit
            if explicit is not None
            else args.source_dir / f"dqn_attacker_{phase}_gc_{args.source_kind}.pt"
        )
    output_dir = args.output_dir.resolve()
    if any(output_dir in path.resolve().parents for path in source_paths.values()):
        raise ValueError("output-dir must not contain source checkpoints")
    if output_dir == args.training_dir.resolve():
        raise ValueError("output-dir must differ from training-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()) and not args.overwrite_output:
        raise ValueError("output-dir must be empty unless --overwrite-output is explicit")
    baseline = {
        phase: common.load_checkpoint(path) for phase, path in source_paths.items()
    }
    baseline_data = baseline["escort"].get("runtime_data_fingerprint")
    if any(
        checkpoint.get("runtime_data_fingerprint") != baseline_data
        for checkpoint in baseline.values()
    ):
        raise ValueError("source checkpoints have mixed runtime data")
    if (
        baseline_data != trainer.runtime_data_fingerprint()
        and not args.allow_data_revision_mismatch
    ):
        raise ValueError("source runtime data differs from the current revision")

    history = json.loads(
        (args.training_dir / "evaluation_history.json").read_text(encoding="utf-8")
    )
    if not history:
        raise ValueError("training evaluation history is empty")
    baseline_metrics = history[0]
    candidate_paths = {
        phase: args.training_dir / f"dqn_attacker_{phase}_gc_best_by_eval.pt"
        for phase in trainer.PHASES
    }
    candidate = {
        phase: common.load_checkpoint(path) for phase, path in candidate_paths.items()
    }
    require_only_carry_escort_movement_changed(
        baseline, candidate, args.allow_facing_changes
    )
    candidate_metrics = candidate["escort"]["evaluation"]
    baseline_score = trainer.joint_movement_selection_score(
        baseline_metrics, baseline_metrics,
        args.max_no_entry_rate, args.max_timeout_rate,
        args.max_quiet_stall_increase, args.max_carrier_death_increase,
        args.max_plant_regression,
    )
    candidate_score = trainer.joint_movement_selection_score(
        candidate_metrics, baseline_metrics,
        args.max_no_entry_rate, args.max_timeout_rate,
        args.max_quiet_stall_increase, args.max_carrier_death_increase,
        args.max_plant_regression,
    )
    selected_score, selected = common.select_candidate(
        baseline_score, {"trained": {"score": candidate_score}}
    )
    safety_regression = trainer.joint_movement_guardrail_violated(
        candidate_metrics,
        baseline_metrics,
        args.max_timeout_rate,
        args.max_quiet_stall_increase,
        args.max_carrier_death_increase,
        args.max_plant_regression,
    )
    if safety_regression:
        selected_score, selected = baseline_score, "baseline"
    paths = source_paths if selected == "baseline" else candidate_paths
    checkpoints = baseline if selected == "baseline" else candidate
    holdout = common.evaluate_combo(
        paths, checkpoints, args.holdout_episodes, args.holdout_seeds
    )
    candidate_holdout_orb_collections = None
    holdout_orb_rejected = False
    if selected == "trained":
        candidate_holdout_orb_collections = int(holdout.get("orb_collections", 0))
        if candidate_holdout_orb_collections < 1:
            # A single evaluation-seed orb pickup is insufficient evidence
            # that the behavior works on unseen rounds. Keep the safe source.
            holdout_orb_rejected = True
            selected_score, selected = baseline_score, "baseline"
            paths, checkpoints = source_paths, baseline
            holdout = common.evaluate_combo(
                paths, checkpoints, args.holdout_episodes, args.holdout_seeds
            )
    fingerprint = trainer.runtime_data_fingerprint()
    bundle = {
        "episode": checkpoints["escort"].get("episode"),
        "selection_method": "carry_escort_joint_movement",
        "selected_branch": selected,
        "evaluation": holdout,
        "models": {},
        "runtime_data_fingerprint": fingerprint,
    }
    for phase in trainer.PHASES:
        payload = dict(checkpoints[phase])
        payload.update(
            evaluation=holdout,
            selection_method="carry_escort_joint_movement",
            selected_branch=selected,
            source_runtime_data_fingerprint=baseline_data,
            runtime_data_fingerprint=fingerprint,
        )
        filename = f"dqn_attacker_{phase}_gc_best_escort_support_ab.pt"
        torch.save(payload, args.output_dir / filename)
        bundle["models"][phase] = filename
    report = {
        "revision": "carry_escort_joint_movement",
        "selection_metrics_source": (
            "evaluation seeds; holdout can veto a trained model with zero orb collections"
        ),
        "baseline_score": list(baseline_score),
        "candidate_score": list(candidate_score),
        "selected_score": list(selected_score),
        "selected": selected,
        "candidate_safety_regression": safety_regression,
        "holdout_orb_rejected": holdout_orb_rejected,
        "candidate_holdout_orb_collections": candidate_holdout_orb_collections,
        "baseline_evaluation": baseline_metrics,
        "candidate_evaluation": candidate_metrics,
        "holdout": holdout,
        "sources": {
            "baseline": {
                phase: common.source_record(source_paths[phase], baseline[phase])
                for phase in trainer.PHASES
            },
            "candidate": {
                phase: common.source_record(candidate_paths[phase], candidate[phase])
                for phase in trainer.PHASES
            },
        },
        "runtime_data_fingerprint": fingerprint,
        "source_runtime_data_fingerprint": baseline_data,
    }
    (args.output_dir / "best_escort_support_ab_bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )
    (args.output_dir / "escort_support_selection_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "holdout_escort_support_ab.json").write_text(
        json.dumps(holdout, indent=2), encoding="utf-8"
    )
    print(
        "[CARRY/ESCORT MOVEMENT SELECTION] "
        + json.dumps(
            {
                "selected": selected,
                "baseline_score": list(baseline_score),
                "candidate_score": list(candidate_score),
            }
        ),
        flush=True,
    )
    print("[CARRY/ESCORT MOVEMENT HOLDOUT] " + json.dumps(holdout), flush=True)


if __name__ == "__main__":
    main()
