"""Current fresh micro family, shared by normal play and the dedicated viewer."""
from pathlib import Path

AI_ROOT=Path(__file__).resolve().parents[1]
MICRO_BC_RUN="attacker_preplant_retrieval_bc_v2"
MICRO_RL_RUN="attacker_preplant_retrieval_v2"
MICRO_INITIALIZATION_SEED=20261025


def active_micro_checkpoint():
    candidates=[AI_ROOT/"data"/run/name for run in (MICRO_RL_RUN,MICRO_BC_RUN)
                for name in ("best.pt","latest.pt")]
    return next((path for path in candidates if path.is_file()),None)


def fresh_micro_policy():
    from .gc3_policy_attacker_site import SiteHistoryPolicy
    return SiteHistoryPolicy.fresh("attacker",MICRO_INITIALIZATION_SEED,navigation=True,retrieval=True)
