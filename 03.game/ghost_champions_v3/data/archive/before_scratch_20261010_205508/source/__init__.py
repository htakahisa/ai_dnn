"""GC v3: independent search and postplant policies, with preserved v2 fallback phases."""
from pathlib import Path
from functools import partial

AI_ROOT = Path(__file__).resolve().parent
V2_SNAPSHOT = AI_ROOT.parent / "ghost_champions_v2/data/version_snapshots/v2_20261010"
BASELINE_CHECKPOINT = V2_SNAPSHOT / "data/attacker_rl_unitstatus_20261009_recovered/latest.pt"
DEFAULT_GUARD_CHECKPOINT = AI_ROOT / "data/guard_v3/best.pt"
DEFAULT_SEARCH_CHECKPOINT = AI_ROOT / "data/defender_search_v3/best.pt"


def build_team_ai():
    from team_ai import DualRoleTeamAI
    from .gc3_learn_defender_search import GhostChampionsV3SearchController
    from .gc3_learn_attacker_guard import GhostChampionsV3AttackerController
    if not DEFAULT_GUARD_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Train GC v3 guard first: {DEFAULT_GUARD_CHECKPOINT}")
    if not DEFAULT_SEARCH_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Train GC v3 defender search first: {DEFAULT_SEARCH_CHECKPOINT}")
    return DualRoleTeamAI("Ghost Champions v3", partial(GhostChampionsV3AttackerController,
                         checkpoint=BASELINE_CHECKPOINT, guard_checkpoint=DEFAULT_GUARD_CHECKPOINT),
                         partial(GhostChampionsV3SearchController, search_checkpoint=DEFAULT_SEARCH_CHECKPOINT))
