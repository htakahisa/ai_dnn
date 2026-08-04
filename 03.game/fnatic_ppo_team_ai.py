from pathlib import Path

from controllers import DefaultDefenderController
from policy_ppo_attacker_controller import PolicyPPOAttackerController
from team_ai import DualRoleTeamAI


def build_fnatic_ppo_team_ai(
    attacker_model_path=(
        "ppo_attacker_checkpoints/"
        "policy_fnatic_attacker_ppo_best.pt"
    ),
    device="auto",
    use_iq_perception=True,
):
    model_path = Path(attacker_model_path)

    return DualRoleTeamAI(
        name="Fnatic PPO",
        attacker_factory=lambda: PolicyPPOAttackerController(
            model_path=model_path,
            device=device,
        ),
        defender_factory=lambda: DefaultDefenderController(),
        use_iq_perception=use_iq_perception,
    )
