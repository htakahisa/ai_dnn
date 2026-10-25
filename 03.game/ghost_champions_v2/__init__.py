"""Ghost Champions v2: attacker tactics and defender attack-site macro."""
import os
from pathlib import Path

# Common settings for competition, roster selection and default live viewing.
# Edit these constants for normal use; environment variables are overrides.
DEFAULT_ATTACKER_MODE = "learned"
DEFAULT_ATTACKER_CHECKPOINT = Path(__file__).resolve().parent / "data/attacker_rl_unitstatus_20261009_recovered/latest.pt"


def resolve_attacker_checkpoint():
    mode = os.environ.get("GC_V2_ATTACKER_MODE", DEFAULT_ATTACKER_MODE)
    if mode not in ("rule", "learned"):
        raise ValueError("GC_V2_ATTACKER_MODE must be 'rule' or 'learned'")
    if mode == "rule":
        return None
    checkpoint = Path(os.environ.get("GC_V2_RL_CHECKPOINT") or DEFAULT_ATTACKER_CHECKPOINT).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GC v2 attacker checkpoint not found: {checkpoint}. "
                                "Set DEFAULT_ATTACKER_CHECKPOINT or explicitly select rule mode.")
    return checkpoint


def build_team_ai():
    from functools import partial
    from team_ai import DualRoleTeamAI
    from .defender_macro_v1.controller import GhostChampionsV2DefenderController
    from .controller import GhostChampionsV2AttackerController
    attacker_factory=GhostChampionsV2AttackerController
    checkpoint = resolve_attacker_checkpoint()
    if checkpoint is not None:
        from .rl.controller import LearnedAttackerController
        attacker_factory=partial(LearnedAttackerController,checkpoint=checkpoint)
    return DualRoleTeamAI(
        name="Ghost Champions v2",
        attacker_factory=attacker_factory,
        defender_factory=GhostChampionsV2DefenderController,
    )
