"""Train an isolated v3 coach for surviving a left-site attack and plant."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from coach_v1.common.checkpoint import CheckpointMetadata
from coach_v1.common.types import Side
from coach_v1.common.versions import ORB_COACH_OBSERVATION_VERSION
from coach_v1.learning_coach_attacker import load_attacker_coach
from coach_v1.models.coach_model import CoachModelConfig
from coach_v1.observation.coach_encoder import (
    COACH_VECTOR_FIELDS, ORB_COACH_GRID_CHANNELS, ORB_COACH_VECTOR_FIELDS,
)
from coach_v1.training.coach_trainer import CoachTrainer


STAGES = ("left_plant", "left_entry", "left_rally", "left_full_round")


def initialize_from_attacker(trainer: CoachTrainer, source: Path | None = None) -> None:
    """Keep old feature weights aligned; initialize only the new inputs to zero."""
    source_policy = load_attacker_coach(source)
    old = source_policy.model.state_dict()
    new = trainer.actor.state_dict()
    if (source_policy.model.config.hidden_channels != trainer.config.hidden_channels
            or source_policy.model.config.hidden_features != trainer.config.hidden_features
            or source_policy.model.config.action_feedback != trainer.config.action_feedback
            or source_policy.model.config.spatial_coordinates != trainer.config.spatial_coordinates):
        raise ValueError("source coach architecture differs beyond observation size")
    for key, value in old.items():
        if key not in {"spatial.0.weight", "input.0.weight", "slot.0.weight"}:
            if new[key].shape != value.shape:
                raise ValueError(f"incompatible source tensor: {key}")
            new[key] = value.clone()
    spatial = new["spatial.0.weight"].clone()
    spatial.zero_()
    spatial[:, :old["spatial.0.weight"].shape[1]] = old["spatial.0.weight"]
    new["spatial.0.weight"] = spatial
    width = trainer.config.hidden_channels
    inputs = new["input.0.weight"].clone()
    inputs.zero_()
    inputs[:, :width] = old["input.0.weight"][:, :width]
    new_fields = {name: index for index, name in enumerate(ORB_COACH_VECTOR_FIELDS)}
    for old_index, name in enumerate(COACH_VECTOR_FIELDS):
        inputs[:, width + new_fields[name]] = old["input.0.weight"][:, width + old_index]
    new["input.0.weight"] = inputs
    slot = new["slot.0.weight"].clone()
    slot.zero_()
    slot[:, :old["slot.0.weight"].shape[1]] = old["slot.0.weight"]
    new["slot.0.weight"] = slot
    trainer.actor.load_state_dict(new, strict=True)


def build_trainer(*, directory: Path, seed: int, source: Path | None = None) -> CoachTrainer:
    actor_path = directory / "latest.pt"
    if actor_path.exists():
        payload = torch.load(actor_path, map_location="cpu", weights_only=False)
        metadata = CheckpointMetadata.from_dict(payload["metadata"])
        config = CoachModelConfig(**metadata.model_config)
    else:
        source_policy = load_attacker_coach(source)
        config = replace(source_policy.model.config,
                         grid_channels=len(ORB_COACH_GRID_CHANNELS),
                         vector_features=len(ORB_COACH_VECTOR_FIELDS),
                         objective_geometry=True, local_spatial=True)
    trainer = CoachTrainer(Side.ATTACKER, seed=seed, config=config,
                           directory=directory,
                           observation_version=ORB_COACH_OBSERVATION_VERSION)
    if actor_path.exists():
        trainer.resume()
    else:
        initialize_from_attacker(trainer, source)
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-ticks", type=int, default=100)
    args = parser.parse_args()
    trainer = build_trainer(directory=args.directory, seed=args.seed,
                            source=args.source)
    history = trainer.fit(episodes=args.episodes, stage=args.stage,
                          max_ticks=args.max_ticks, save_rollouts=False)
    print(history[-1])


if __name__ == "__main__":
    main()
