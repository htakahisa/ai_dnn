"""Evaluate one saved joint positioning checkpoint without changing its weights."""
import argparse
import json
from pathlib import Path

import torch

import train_attacker_gc_real_curriculum as trainer


def build_policies(checkpoints):
    """Build the current runtime architectures and expand older checkpoints."""
    policies = {
        "carry": trainer.runtime.AttackerCarryDuelingDQN(
            obs_dim=trainer.runtime.FACING_HEAD_OBS_DIM,
            action_dim=trainer.runtime.ACTION_DIM,
        ),
        "escort": trainer.escort_runtime.DuelingQNetwork(
            trainer.escort_runtime.FACING_HEAD_OBS_DIM,
            trainer.escort_runtime.N_ACTIONS,
        ),
        "guard": trainer.guard_runtime.AttackerGuardDuelingDQN(
            obs_dim=trainer.guard_runtime.ULTIMATE_CONTEXT_OBS_DIM,
            action_dim=trainer.guard_runtime.ACTION_DIM,
        ),
    }
    for phase in trainer.PHASES:
        policy = policies[phase]
        state = trainer.expand_policy_state(
            checkpoints[phase],
            policy.feature[0].in_features,
            policy.advantage_head[-1].out_features,
        )
        incompatible = policy.load_state_dict(state, strict=False)
        allowed_missing = (
            phase in trainer.FACING_PHASES
            and all(trainer.is_facing_parameter(key) for key in incompatible.missing_keys)
        )
        if (incompatible.missing_keys and not allowed_missing) or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{phase} checkpoint keys mismatch: "
                f"missing={list(incompatible.missing_keys)}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )
        if phase in trainer.FACING_PHASES:
            policy.facing_head_version = int(
                checkpoints[phase].get("facing_head_version", 0)
            )
    return policies


def validate_runtime_data(checkpoints, allow_mismatch=False):
    """Reject silent evaluation against roster data changed after training."""
    recorded_by_phase = {
        phase: checkpoint.get("runtime_data_fingerprint")
        for phase, checkpoint in checkpoints.items()
    }
    recorded = [
        value for value in recorded_by_phase.values() if value is not None
    ]
    if not recorded:
        return None, trainer.runtime_data_fingerprint()
    if len(recorded) != len(checkpoints):
        raise ValueError("checkpoint set contains mixed runtime-data revisions")
    if any(value != recorded[0] for value in recorded[1:]):
        raise ValueError("checkpoint set contains mixed runtime-data revisions")
    current = trainer.runtime_data_fingerprint()
    if recorded[0] != current and not allow_mismatch:
        changed = sorted(
            name
            for name in set(recorded[0]) | set(current)
            if recorded[0].get(name) != current.get(name)
        )
        raise ValueError(
            "runtime data differs from the training revision: "
            f"{changed}; pass --allow-data-revision-mismatch only for an "
            "intentional cross-revision evaluation"
        )
    return recorded[0], current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=(
            "latest",
            "best_by_eval",
            "best_phase_facing",
            "best_phase_facing_ab",
            "best_carry_movement_ab",
            "best_escort_support_ab",
        ),
        default="best_by_eval",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-data-revision-mismatch", action="store_true")
    parser.add_argument(
        "--escort-model",
        type=Path,
        help="Use this Escort checkpoint for a diagnostic mixed-episode comparison",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    torch.set_num_threads(1)
    sources = {p: args.models_dir / f"dqn_attacker_{p}_gc_{args.kind}.pt" for p in trainer.PHASES}
    if args.escort_model is not None:
        sources["escort"] = args.escort_model
    checkpoints = {p: torch.load(path, map_location="cpu", weights_only=False)
                   for p, path in sources.items()}
    if args.escort_model is None and len({c.get("episode") for c in checkpoints.values()}) != 1:
        raise ValueError("checkpoint set is not synchronized")
    recorded_data, current_data = validate_runtime_data(
        checkpoints, args.allow_data_revision_mismatch
    )
    policies = build_policies(checkpoints)
    session = trainer.CurriculumSession(sources, policies, 0.99)
    for phase in trainer.PHASES:
        session.controllers[phase].positioning_version = checkpoints[phase].get("positioning_version", 0)
        if phase in trainer.FACING_PHASES:
            session.controllers[phase].facing_head_enabled = (
                int(checkpoints[phase].get("facing_head_version", 0)) >= 1
            )
    report = trainer.evaluate_multi(session, args.episodes, args.seeds, (100, 60, 40))
    report["checkpoint_episode"] = checkpoints["carry"].get("episode")
    report["checkpoint_episodes_by_phase"] = {
        phase: checkpoints[phase].get("episode") for phase in trainer.PHASES
    }
    report["checkpoint_paths_by_phase"] = {
        phase: str(sources[phase]) for phase in trainer.PHASES
    }
    report["training_runtime_data_fingerprint"] = recorded_data
    report["evaluation_runtime_data_fingerprint"] = current_data
    content = json.dumps(report, ensure_ascii=False, indent=2)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
