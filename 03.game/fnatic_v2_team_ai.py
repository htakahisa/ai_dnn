from __future__ import annotations

from pathlib import Path

from policy_defender_controller import PolicyDefenderController
from policy_ppo_attacker_controller import PolicyPPOAttackerController
from team_ai import DualRoleTeamAI


DEFAULT_ATTACKER_MODEL = Path(
    "ppo_attacker_checkpoints/"
    "policy_fnatic_attacker_ppo_best.pt"
)
DEFAULT_DEFENDER_MODEL = Path(
    "policy_fnatic_defender_dagger_final.pt"
)


def build_fnatic_v2_team_ai(
    attacker_model_path: str | Path = DEFAULT_ATTACKER_MODEL,
    defender_model_path: str | Path = DEFAULT_DEFENDER_MODEL,
    *,
    device: str = "auto",
    use_iq_perception: bool = True,
) -> DualRoleTeamAI:
    """
    Fnatic v2を生成する。

    Attacker:
        PPO追加学習済みモデル

    Defender:
        現在のBC/DAgger模倣学習モデル

    攻守交代時はDualRoleTeamAIが対応Controllerへ自動切替する。
    """
    attacker_model_path = Path(attacker_model_path)
    defender_model_path = Path(defender_model_path)

    return DualRoleTeamAI(
        name="Fnatic v2",
        attacker_factory=lambda: PolicyPPOAttackerController(
            model_path=attacker_model_path,
            device=device,
        ),
        defender_factory=lambda: PolicyDefenderController(
            model_path=defender_model_path,
            device=device,
        ),
        use_iq_perception=use_iq_perception,
    )


def create_fnatic_v2(
    *,
    device: str = "auto",
    use_iq_perception: bool = True,
) -> DualRoleTeamAI:
    """引数を省略してFnatic v2を生成する簡易関数。"""
    return build_fnatic_v2_team_ai(
        device=device,
        use_iq_perception=use_iq_perception,
    )
