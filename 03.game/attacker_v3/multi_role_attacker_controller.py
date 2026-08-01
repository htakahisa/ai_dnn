# attacker_v2/multi_role_attacker_controller.py
"""
run_game.py側は1つのコントローラーとしてこれを受け取り、内部でキャラの状態に応じて
キャリアーモデル/護衛モデルへ処理を振り分ける。run_game.py自体への変更を避けるための窓口。
"""

import sys
import numpy as np
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from controllers import BaseController, DefaultAttackerController
from learning_attacker_carry import LearningAttackerCarryController
from learning_attacker_escort import LearningAttackerEscortController
from learning_attacker_retrieve import LearningAttackerRetrieveController
from learning_attacker_guard import LearningAttackerGuardController


class MultiRoleAttackerController(BaseController):
    """attacker側の窓口となる単一コントローラー。
    内部でキャリアーモデル・護衛モデルを保持し、キャラごとに使い分ける。
    run_game.py からは通常のコントローラー1つとして扱われる(isinstance判定にはこのクラスを追加する)。"""

    def __init__(
        self,
        carry_model_path="attacker_v3/data/attacker_carry_data/dqn_attacker_carry_best_by_eval.pt",
        escort_model_path="attacker_v3/data/attacker_escort_data/dqn_attacker_escort_best_by_eval.pt",
        retrieve_model_path="attacker_v3/data/attacker_retrieve_data/dqn_attacker_retrieve_best_by_eval.pt",
        guard_model_path="attacker_v3/data/attacker_guard_data/dqn_attacker_guard_best_by_eval.pt",
        greedy=False,
    ):
        super().__init__()
        self.carry_controller = LearningAttackerCarryController(model_path=carry_model_path, greedy=greedy)
        self.escort_controller = LearningAttackerEscortController(model_path=escort_model_path, greedy=greedy)
        self.retrieve_controller = LearningAttackerRetrieveController(model_path=retrieve_model_path, greedy=greedy)
        self.guard_controller = LearningAttackerGuardController(model_path=guard_model_path, greedy=greedy)

        # 💡追加: チーム内で「サイト内で誰かが既にアビリティを使用したか」を共有する状態。
        # ラウンドごとにreset_roundでクリアする。
        self.site_ability_used_by_team = False

    def _build_or_fallback(self, model_path, param_name):
        if model_path is not None:
            raise NotImplementedError(f"{param_name} に対応するモデルは未実装です。Noneのままにしてください。")
        return DefaultAttackerController()

    def reset_round(self):
        for ctrl in (self.carry_controller, self.escort_controller, self.retrieve_controller, self.guard_controller):
            if hasattr(ctrl, "reset_round"):
                ctrl.reset_round()
        self.site_ability_used_by_team = False  # 💡追加

    def decide_move(self, char, game_state):
        if game_state.get("is_planted"):
            return self.guard_controller.decide_move(char, game_state)

        if char.has_spike:
            result = self.carry_controller.decide_move(char, game_state)
            self._maybe_mark_site_ability_used(char, result, self.carry_controller)
            return result

        chars = game_state.get("chars", [])
        holder = next((c for c in chars if c.is_alive and c.team == "A" and c.has_spike), None)
        if holder is None:
            return self.retrieve_controller.decide_move(char, game_state)

        # 💡追加: escortには使用前に共有フラグを渡す
        self.escort_controller.site_ability_used_by_teammate = self.site_ability_used_by_team
        result = self.escort_controller.decide_move(char, game_state)
        self._maybe_mark_site_ability_used(char, result, self.escort_controller)
        return result

    def _maybe_mark_site_ability_used(self, char, result, sub_controller):
        """アビリティがサイト付近で使用されたかを判定し、共有フラグを更新する。
        💡簡略化: 狙った方向の精密な判定はせず、「使用者がサイトに一定距離まで近づいていたか」
        だけで判定する(carry/escort双方のdist_mapがこの時点で正しく更新されている前提)。"""
        _, action_type = result
        if action_type != "ABILITY":
            return
        if not hasattr(sub_controller, "dist_map"):
            return
        r, c = char.pos
        dist = sub_controller.dist_map[r, c]
        if np.isfinite(dist) and dist <= SITE_APPROACH_DIST_THRESHOLD:
            self.site_ability_used_by_team = True