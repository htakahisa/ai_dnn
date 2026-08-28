# attacker_v2/multi_role_defender_controller.py
"""
run_game.py側は1つのコントローラーとしてこれを受け取り、内部でキャラの状態(is_planted)
に応じてsearchモデル/retakeモデルへ処理を振り分ける。run_game.py自体への変更を避ける
ための窓口。multi_role_attacker_controller.py のDefender版。

ルーティング方針:
    - is_planted == False (プラント前) -> LearningDefenderSearchController
    - is_planted == True  (プラント後・退がり/リテイク) -> LearningDefenderRetakeController

search / retake いずれも「Defenderチーム全員(最大5人)で1インスタンスを共有する」
設計になっているため(team_memory・味方アビリティ使用状況の追跡のため)、
この窓口クラスでも各フェーズにつき1インスタンスだけを保持し、全キャラクターの
decide_move呼び出しを同じインスタンスへ中継する。個別インスタンス化はしない。
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from controllers import BaseController, DefaultDefenderController
from tv2_learning_defender_retake_touyama import LearningDefenderRetakeTouyamaController
from tv2_learning_defender_search_touyama import LearningDefenderSearchTouyamaController


class Tv2TouyamaDefenderController(BaseController):
    """defender側の窓口となる単一コントローラー。
    内部でsearchモデル・retakeモデルを保持し、is_plantedに応じて使い分ける。
    run_game.py からは通常のコントローラー1つとして扱われる(isinstance判定にはこのクラスを追加する)。"""

    def __init__(
        self,
        search_model_path="touyama_v2/data/defender_search_touyama_data/dqn_defender_search_touyama_best_by_eval.pt",
        retake_model_path="touyama_v2/data/defender_retake_touyama_data/dqn_defender_retake_touyama_best_by_eval.pt",
        greedy=True,
    ):
        super().__init__()
        self.search_controller = LearningDefenderSearchTouyamaController(
            model_path=search_model_path, greedy=greedy
        )
        self.retake_controller = LearningDefenderRetakeTouyamaController(
            model_path=retake_model_path, greedy=greedy
        )

    def _build_or_fallback(self, model_path, param_name):
        if model_path is not None:
            raise NotImplementedError(f"{param_name} に対応するモデルは未実装です。Noneのままにしてください。")
        return DefaultDefenderController()

    def reset_round(self):
        for ctrl in (self.search_controller, self.retake_controller):
            if hasattr(ctrl, "reset_round"):
                ctrl.reset_round()

    def decide_move(self, char, game_state):
        if game_state.get("is_planted"):
            return self.retake_controller.decide_move(char, game_state)

        return self.search_controller.decide_move(char, game_state)