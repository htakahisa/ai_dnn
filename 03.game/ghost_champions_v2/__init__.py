"""Ghost Champions v2: attacker tactics and defender attack-site macro."""


def build_team_ai():
    import os
    from functools import partial
    from team_ai import DualRoleTeamAI
    from .defender_macro_v1.controller import GhostChampionsV2DefenderController
    from .controller import GhostChampionsV2AttackerController
    attacker_factory=GhostChampionsV2AttackerController
    if os.environ.get("GC_V2_RL_CHECKPOINT"):
        from .rl.controller import LearnedAttackerController
        attacker_factory=partial(LearnedAttackerController,checkpoint=os.environ["GC_V2_RL_CHECKPOINT"])
    return DualRoleTeamAI(
        name="Ghost Champions v2",
        attacker_factory=attacker_factory,
        defender_factory=GhostChampionsV2DefenderController,
    )
