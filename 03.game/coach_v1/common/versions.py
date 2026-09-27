"""Compatibility versions frozen by coach_v1/03.DESIGN.md."""

from dataclasses import dataclass

from .types import ModelFamily


CHECKPOINT_SCHEMA_VERSION = "coach-checkpoint-v1"
COACH_OBSERVATION_VERSION = "coach-observation-v1"
COACH_ACTION_VERSION = "coach-action-v1"
CHARACTER_OBSERVATION_VERSION = "character-observation-v1"
CHARACTER_ACTION_VERSION = "character-action-v1"


@dataclass(frozen=True)
class InterfaceVersions:
    observation: str
    action: str


def interface_versions_for(family: ModelFamily) -> InterfaceVersions:
    if family is ModelFamily.COACH:
        return InterfaceVersions(COACH_OBSERVATION_VERSION, COACH_ACTION_VERSION)
    if family is ModelFamily.CHARACTER:
        return InterfaceVersions(
            CHARACTER_OBSERVATION_VERSION, CHARACTER_ACTION_VERSION
        )
    raise ValueError(f"unsupported model family: {family!r}")
