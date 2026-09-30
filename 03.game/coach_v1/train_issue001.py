"""Fine-tune isolated, position-aware Issue 001 coach candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from coach_v1.common.types import Side
from coach_v1.models.coach_model import CoachModelConfig
from coach_v1.training.attacker_imitation import fit_imitation as fit_attacker
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.defender_imitation import fit_balanced_imitation


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "checkpoints/experiments/task16_watch_added/coach"
TARGET = ROOT / "checkpoints/experiments/issue001_distance/coach"


def prepare(side: Side, directory: Path) -> CoachTrainer:
    """Copy old weights into a new, explicitly incompatible config."""
    source = SOURCE / side.value
    config = CoachModelConfig(spatial_coordinates=True, objective_geometry=True,
                              local_spatial=True)
    trainer = CoachTrainer(side, seed=11, config=config, directory=directory)
    if (directory / "latest.pt").exists():
        trainer.resume()
        return trainer
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("candidate directory has files but no latest.pt")
    actor = torch.load(source / "latest.pt", map_location="cpu", weights_only=False)
    training = torch.load(source / "training_latest.pt", map_location="cpu", weights_only=False)
    state = dict(actor["model_state_dict"])
    key = "slot.0.weight"
    expanded = trainer.actor.state_dict()[key].clone()
    expanded.zero_()
    expanded[:, :state[key].shape[1]] = state[key]
    state[key] = expanded
    trainer.actor.load_state_dict(state, strict=True)
    trainer.critic.load_state_dict(training["critic_state_dict"], strict=True)
    # The widened slot projection needs a fresh optimizer state.
    trainer.training_step = int(actor["metadata"]["training_step"])
    trainer.episode = int(training["episode"])
    trainer.history = list(training["history"])
    trainer.save()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("attacker", "defender"), required=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--retake-only", action="store_true")
    parser.add_argument("--carrier-move-weight", type=float, default=1.0)
    parser.add_argument("--retake-move-weight", type=float, default=1.0)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if (args.cycles <= 0 or args.carrier_move_weight < 1
            or args.retake_move_weight < 1):
        parser.error("cycles must be positive and movement weights at least one")
    torch.set_num_threads(1)
    side = Side(args.side)
    directory = args.directory or TARGET / side.value
    trainer = prepare(side, directory)
    if side is Side.ATTACKER:
        for _ in range(args.cycles):
            for stage, ticks in (("rally", 80), ("entry", 40),
                                 ("plant", 18), ("full_round", 120)):
                fit_attacker(trainer, episodes=1, max_ticks=ticks,
                             on_policy=stage == "full_round", stage=stage,
                             carrier_move_weight=args.carrier_move_weight)
    else:
        for _ in range(args.cycles):
            stage_ticks = ({"group_up": 45, "ability_retake": 45,
                            "defuse_escort": 35} if args.retake_only else
                           {"initial_setup": 20, "group_up": 45,
                            "ability_retake": 45, "defuse_escort": 35,
                            "full_round": 120})
            fit_balanced_imitation(
                trainer, cycles=1,
                stage_ticks=stage_ticks,
                samples_per_bucket=32,
                paired_sites=True,
                retake_move_weight=args.retake_move_weight,
            )
    print(f"{side.value}: episode={trainer.episode} step={trainer.training_step}"
          f" checkpoint={directory / 'latest.pt'}")


if __name__ == "__main__":
    main()
