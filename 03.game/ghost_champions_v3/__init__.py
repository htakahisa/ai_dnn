"""Independent GC v3. Missing trained models use freshly initialized v3 weights."""
from pathlib import Path
from functools import partial

AI_ROOT = Path(__file__).resolve().parent
ATTACKER_SEED = 20261011
DEFENDER_SEED = 20261012
USE_ATTACKER_MACRO = False
DEFAULT_GUARD_CHECKPOINT = AI_ROOT / "data/attacker_postplant_fixed_v1/best.pt"
DEFAULT_PREPLANT_CHECKPOINT = AI_ROOT / "data/attacker_preplant_rl_history_v2/best.pt"
BC_PREPLANT_CHECKPOINT = AI_ROOT / "data/attacker_preplant_bc_history_v2/best.pt"
OMOKO_RL_CHECKPOINT = AI_ROOT / "data/attacker_preplant_omoko_rl_site_v1/best.pt"
RESOURCE_RL_CHECKPOINT = AI_ROOT / "data/attacker_preplant_omoko_rl_resources_v2/best.pt"
HANDOFF_RL_CHECKPOINT = AI_ROOT / "data/attacker_preplant_omoko_rl_handoff_v3/best.pt"
MACRO_CHECKPOINT = AI_ROOT / "macro_v1/data/attacker_macro_v1/best.pt"
MAJOR_MACRO_CHECKPOINT = AI_ROOT / "macro_v1/data/attacker_macro_major_v1/best.pt"
OMOKO_BC_CHECKPOINT = AI_ROOT / "data/attacker_preplant_omoko_bc_site_v1/best.pt"
DEFAULT_SEARCH_CHECKPOINT = AI_ROOT / "data/defender_search_fixed_v1/best.pt"
INITIAL_GUARD_CHECKPOINT = AI_ROOT / "data/attacker_postplant_fixed_v1/initial.pt"
INITIAL_SEARCH_CHECKPOINT = AI_ROOT / "data/defender_search_fixed_v1/initial.pt"
# Retained models are inference-only fallbacks until the fixed runs exist.
GUARD_FALLBACK_CHECKPOINTS = tuple(AI_ROOT/path for path in (
    "data/attacker_postplant_reward_v1/best.pt", "data/attacker_postplant_scratch/best.pt",
    "data/attacker_postplant_scratch/initial.pt"))
SEARCH_FALLBACK_CHECKPOINTS = tuple(AI_ROOT/path for path in (
    "data/defender_search_scratch/best.pt", "data/defender_search_scratch/initial.pt"))


def _checkpoint(trained, initial, fallbacks=()):
    return next((path for path in (trained, trained.with_name("latest.pt"), initial, *fallbacks)
                 if path.is_file()), None)


def build_team_ai():
    from team_ai import DualRoleTeamAI
    from .preplant.gc3_learn_attacker_preplant import TwoPhaseAttackerController
    attacker_type = TwoPhaseAttackerController
    macro_settings = {}
    macro_checkpoint = MAJOR_MACRO_CHECKPOINT if MAJOR_MACRO_CHECKPOINT.is_file() else MACRO_CHECKPOINT
    if USE_ATTACKER_MACRO and macro_checkpoint.is_file():
        from .macro_v1.gc3_learn_attacker_macro import MacroAttackerController
        attacker_type = MacroAttackerController
        macro_settings = dict(macro_checkpoint=macro_checkpoint)
    if USE_ATTACKER_MACRO:
        attacker_settings=dict(preplant_checkpoint=next((path for path in (HANDOFF_RL_CHECKPOINT,
            HANDOFF_RL_CHECKPOINT.with_name("latest.pt"),RESOURCE_RL_CHECKPOINT,OMOKO_RL_CHECKPOINT,
            OMOKO_BC_CHECKPOINT,DEFAULT_PREPLANT_CHECKPOINT) if path.is_file()),None))
    else:
        from .preplant.gc3_checkpoint_attacker_micro import active_micro_checkpoint,fresh_micro_policy
        checkpoint=active_micro_checkpoint()
        attacker_settings=dict(preplant_checkpoint=checkpoint)
        if checkpoint is None:attacker_settings["preplant_policy"]=fresh_micro_policy()
    from .gc3_learn_defender_search import GhostChampionsV3SearchController
    from .preplant.gc3_learn_attacker_preplant import HISTORY_LOG_ROOT, SAVE_MATCH_HISTORY
    from .preplant.gc3_observation_attacker_history import MatchHistory
    from .preplant.gc3_history_attacker_match import ObservedDefender
    history = MatchHistory.live_match(HISTORY_LOG_ROOT) if SAVE_MATCH_HISTORY else MatchHistory()
    history.live = True
    return DualRoleTeamAI("Ghost Champions v3",
        partial(attacker_type, **macro_settings,
                history=history,
                **attacker_settings,
                guard_checkpoint=_checkpoint(DEFAULT_GUARD_CHECKPOINT, INITIAL_GUARD_CHECKPOINT, GUARD_FALLBACK_CHECKPOINTS)),
        lambda: ObservedDefender(GhostChampionsV3SearchController(
            checkpoint=_checkpoint(DEFAULT_SEARCH_CHECKPOINT, INITIAL_SEARCH_CHECKPOINT, SEARCH_FALLBACK_CHECKPOINTS)), history))
