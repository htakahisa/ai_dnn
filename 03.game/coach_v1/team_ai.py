"""Runtime factory for the trained coach_v1 team.

This module is the only composition point that joins the attacker coach,
defender coach, and five character policies.  Gameplay and side switching
remain owned by the existing ``DualRoleTeamAI`` / ``VisualFPSBattle`` APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from team_ai import DualRoleTeamAI

from coach_v1.common.constants import CHARACTER_CHECKPOINT_IDS
from coach_v1.common.types import Side
from coach_v1.coordinator import TeamExecutionCoordinator
from coach_v1.learning_character_gongon import GongonPolicy
from coach_v1.learning_character_gonta import GontaPolicy
from coach_v1.learning_character_gorimaru import GorimaruPolicy
from coach_v1.learning_character_kunta import KuntaPolicy
from coach_v1.learning_character_kurimaru import KurimaruPolicy
from coach_v1.learning_coach_attacker import load_attacker_coach
from coach_v1.learning_coach_defender import load_defender_coach


_CHARACTER_POLICY_TYPES = (
    GorimaruPolicy,
    GongonPolicy,
    GontaPolicy,
    KuntaPolicy,
    KurimaruPolicy,
)


def _character_actors(
    checkpoint_paths: Mapping[str, Path] | None,
    *,
    device: str,
):
    paths = dict(checkpoint_paths or {})
    unknown = set(paths) - set(CHARACTER_CHECKPOINT_IDS)
    if unknown:
        raise ValueError(f"unknown character checkpoint ids: {sorted(unknown)}")
    return {
        slot: policy_type(paths.get(checkpoint_id), device=device)
        for slot, (checkpoint_id, policy_type) in enumerate(
            zip(CHARACTER_CHECKPOINT_IDS, _CHARACTER_POLICY_TYPES)
        )
    }


def build_coach_v1_team(
    *,
    name: str = "coach_v1",
    attacker_checkpoint: Path | None = None,
    defender_checkpoint: Path | None = None,
    character_checkpoints: Mapping[str, Path] | None = None,
    attacker_coach=None,
    defender_coach=None,
    device: str = "cpu",
) -> DualRoleTeamAI:
    """Build the trained dual-role team with strict side-specific coaches.

    Each side receives its own coordinator, belief memory, recurrent coach
    state, and character policy instances.  ``DualRoleTeamAI`` selects the
    correct coordinator after every normal or overtime side swap.
    """

    def controller(side: Side) -> TeamExecutionCoordinator:
        if side is Side.ATTACKER:
            coach = (attacker_coach if attacker_coach is not None else
                     load_attacker_coach(attacker_checkpoint, device=device))
        else:
            coach = (defender_coach if defender_coach is not None else
                     load_defender_coach(defender_checkpoint, device=device))
        return TeamExecutionCoordinator(
            side,
            coach,
            _character_actors(character_checkpoints, device=device),
        )

    return DualRoleTeamAI(
        name=name,
        attacker_factory=lambda: controller(Side.ATTACKER),
        defender_factory=lambda: controller(Side.DEFENDER),
        # TeamExecutionCoordinator owns the legal shared-team sensor boundary.
        use_iq_perception=False,
    )
