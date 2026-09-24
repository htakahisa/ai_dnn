"""Checkpoint metadata construction and strict compatibility validation.

Saving/loading tensors remains the responsibility of the matching trainer and
learning module.  Keeping this module independent of torch makes the common
contract cheap to import and reusable by every model.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .constants import CHARACTER_CHECKPOINT_IDS, FIXED_ROSTER_NAMES
from .types import ModelFamily, ModelTarget, Side
from .versions import CHECKPOINT_SCHEMA_VERSION, interface_versions_for


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "observation_version",
        "action_version",
        "map_hash",
        "watch_points_hash",
        "roster",
        "model_family",
        "target_id",
        "model_config",
        "training_seed",
        "training_step",
        "created_at_utc",
    }
)


class CheckpointCompatibilityError(ValueError):
    """Raised when checkpoint metadata is not safe for the current runtime."""


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _validate_target(target: ModelTarget) -> None:
    if target.family is ModelFamily.COACH:
        if target.target_id not in {side.value for side in Side}:
            raise ValueError(f"invalid coach side: {target.target_id!r}")
        return
    if target.family is ModelFamily.CHARACTER:
        if target.target_id not in CHARACTER_CHECKPOINT_IDS:
            raise ValueError(f"invalid character checkpoint id: {target.target_id!r}")
        return
    raise ValueError(f"unsupported model family: {target.family!r}")


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: str
    observation_version: str
    action_version: str
    map_hash: str
    watch_points_hash: str
    roster: Tuple[str, ...]
    model_family: ModelFamily
    target_id: str
    model_config: Mapping[str, Any]
    training_seed: int
    training_step: int
    created_at_utc: str

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCompatibilityError(
                f"unsupported checkpoint schema: {self.schema_version!r}"
            )
        current_versions = interface_versions_for(self.model_family)
        if self.observation_version != current_versions.observation:
            raise CheckpointCompatibilityError(
                f"unsupported observation version: {self.observation_version!r}"
            )
        if self.action_version != current_versions.action:
            raise CheckpointCompatibilityError(
                f"unsupported action version: {self.action_version!r}"
            )
        _require_sha256("map_hash", self.map_hash)
        _require_sha256("watch_points_hash", self.watch_points_hash)
        if tuple(self.roster) != FIXED_ROSTER_NAMES:
            raise ValueError("roster must match the fixed coach_v1 slot order")
        _validate_target(ModelTarget(self.model_family, self.target_id))
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            raise TypeError("training_seed must be int")
        if (
            isinstance(self.training_step, bool)
            or not isinstance(self.training_step, int)
            or self.training_step < 0
        ):
            raise ValueError("training_step must be a non-negative int")
        if not isinstance(self.model_config, Mapping):
            raise TypeError("model_config must be a mapping")
        try:
            parsed_time = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("created_at_utc must be an ISO-8601 timestamp") from exc
        if parsed_time.tzinfo is None or parsed_time.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("created_at_utc must use UTC")

    @property
    def target(self) -> ModelTarget:
        return ModelTarget(self.model_family, self.target_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_version": self.observation_version,
            "action_version": self.action_version,
            "map_hash": self.map_hash,
            "watch_points_hash": self.watch_points_hash,
            "roster": list(self.roster),
            "model_family": self.model_family.value,
            "target_id": self.target_id,
            "model_config": dict(self.model_config),
            "training_seed": self.training_seed,
            "training_step": self.training_step,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointMetadata":
        if not isinstance(value, Mapping):
            raise TypeError("checkpoint metadata must be a mapping")
        keys = frozenset(value.keys())
        if keys != _METADATA_KEYS:
            missing = sorted(_METADATA_KEYS - keys)
            extra = sorted(keys - _METADATA_KEYS)
            raise ValueError(
                f"invalid checkpoint metadata keys: missing={missing}, extra={extra}"
            )
        try:
            family = ModelFamily(value["model_family"])
            roster = tuple(value["roster"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid checkpoint model_family or roster") from exc
        return cls(
            schema_version=value["schema_version"],
            observation_version=value["observation_version"],
            action_version=value["action_version"],
            map_hash=value["map_hash"],
            watch_points_hash=value["watch_points_hash"],
            roster=roster,
            model_family=family,
            target_id=value["target_id"],
            model_config=dict(value["model_config"]),
            training_seed=value["training_seed"],
            training_step=value["training_step"],
            created_at_utc=value["created_at_utc"],
        )


def build_checkpoint_metadata(
    *,
    target: ModelTarget,
    map_hash: str,
    watch_points_hash: str,
    model_config: Mapping[str, Any],
    training_seed: int,
    training_step: int,
    created_at_utc: Optional[str] = None,
) -> CheckpointMetadata:
    _validate_target(target)
    versions = interface_versions_for(target.family)
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return CheckpointMetadata(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        observation_version=versions.observation,
        action_version=versions.action,
        map_hash=map_hash,
        watch_points_hash=watch_points_hash,
        roster=FIXED_ROSTER_NAMES,
        model_family=target.family,
        target_id=target.target_id,
        model_config=dict(model_config),
        training_seed=training_seed,
        training_step=training_step,
        created_at_utc=timestamp,
    )


def validate_checkpoint_compatibility(
    loaded: CheckpointMetadata, expected: CheckpointMetadata
) -> None:
    """Require exact inference compatibility; training provenance may differ."""

    compatibility_fields = (
        "schema_version",
        "observation_version",
        "action_version",
        "map_hash",
        "watch_points_hash",
        "roster",
        "model_family",
        "target_id",
        "model_config",
    )
    mismatches = [
        field
        for field in compatibility_fields
        if getattr(loaded, field) != getattr(expected, field)
    ]
    if mismatches:
        raise CheckpointCompatibilityError(
            "incompatible checkpoint metadata fields: " + ", ".join(mismatches)
        )


def build_checkpoint_payload(
    *,
    metadata: CheckpointMetadata,
    model_state_dict: Mapping[str, Any],
    optimizer_state_dict: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the minimum serializable payload required by the v1 contract."""

    return {
        "metadata": metadata.to_dict(),
        "model_state_dict": dict(model_state_dict),
        "optimizer_state_dict": (
            None if optimizer_state_dict is None else dict(optimizer_state_dict)
        ),
    }
