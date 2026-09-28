"""Actor-only loader matching train_coach.py and coordinator's coach contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from coach_v1.common.checkpoint import (
    CheckpointMetadata, build_checkpoint_metadata, validate_checkpoint_compatibility,
)
from coach_v1.common.constants import COACH_CHECKPOINT_PATHS
from coach_v1.common.types import ModelTarget, Side
from coach_v1.models.coach_model import CoachActorModel, CoachModelConfig, CoachPolicy
from coach_v1.observation.coach_encoder import CoachObservationEncoder


def load_coach_policy(side: Side, path: Path | None = None, *, device: str = "cpu") -> CoachPolicy:
    if not isinstance(side, Side):
        raise ValueError("coach side must be attacker or defender")
    checkpoint = Path(path) if path is not None else COACH_CHECKPOINT_PATHS[side.value] / "latest.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = CheckpointMetadata.from_dict(payload["metadata"])
    config_values = dict(metadata.model_config)
    config_values.setdefault("action_feedback", False)  # checkpoints made before feedback input
    config = CoachModelConfig(**config_values)
    encoder = CoachObservationEncoder(version=metadata.observation_version)
    expected = build_checkpoint_metadata(
        target=ModelTarget.coach(side), map_hash=encoder.map_hash,
        watch_points_hash=encoder.watch_points_hash,
        model_config=config.to_dict(), training_seed=metadata.training_seed,
        training_step=0,
    )
    validate_checkpoint_compatibility(
        metadata, replace(expected, observation_version=encoder.version)
    )
    encoder.validate_checkpoint(metadata, side=side)
    if payload.get("optimizer_state_dict") is not None or "critic_state_dict" in payload:
        raise ValueError("actor checkpoint contains training-only state")
    model = CoachActorModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return CoachPolicy(model, encoder, device=device)
