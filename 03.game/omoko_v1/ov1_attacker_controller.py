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

from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import FixedAbilityAdvisor, MAPS
from omoko_v1.ability_advisor_v1.ov1_advisor_attacker_context import (
    policy_path, policy_digest, scenario, defending_opponent,
)
from controllers import BaseController, DefaultAttackerController
from ov1_learning_attacker_carry import Ov1LearningAttackerCarryController
from ov1_learning_attacker_escort import Ov1LearningAttackerEscortController
from ov1_learning_attacker_retrieve import Ov1LearningAttackerRetrieveController
from ov1_learning_attacker_guard import Ov1LearningAttackerGuardController
from ov1_inference_damage_avoidance import escape_destruction_area, avoid_destruction_entry
from ov1_inference_team_tactics import apply_shared_facing, tactical_state, known_enemies, shot_clear


class Ov1AttackerController(BaseController):
    """attacker側の窓口となる単一コントローラー。
    内部でキャリアーモデル・護衛モデルを保持し、キャラごとに使い分ける。
    run_game.py からは通常のコントローラー1つとして扱われる(isinstance判定にはこのクラスを追加する)。"""

    requires_public_effects = True
    requires_exact_team_tactics = True

    def __init__(
        self,
        carry_model_path="omoko_v1/data/attacker_carry_data/dqn_attacker_carry_best_by_eval.pt",
        escort_model_path="omoko_v1/data/attacker_escort_data/dqn_attacker_escort_best_by_eval.pt",
        retrieve_model_path="omoko_v1/data/attacker_retrieve_data/dqn_attacker_retrieve_best_by_eval.pt",
        guard_model_path="omoko_v1/data/attacker_guard_data/dqn_attacker_guard_best_by_eval.pt",
        greedy=False,
        device=None,
        ability_advisor=None,
    ):
        super().__init__()
        paths = dict(carry=carry_model_path, escort=escort_model_path,
                     retrieve=retrieve_model_path, guard=guard_model_path)
        carry_model_path, escort_model_path, retrieve_model_path, guard_model_path = (
            policy_path(paths[role]) for role in ('carry', 'escort', 'retrieve', 'guard'))
        self.carry_controller = Ov1LearningAttackerCarryController(model_path=carry_model_path, greedy=greedy, device=device)
        self.escort_controller = Ov1LearningAttackerEscortController(model_path=escort_model_path, greedy=greedy, device=device)
        self.retrieve_controller = Ov1LearningAttackerRetrieveController(model_path=retrieve_model_path, greedy=greedy, device=device)
        self.guard_controller = Ov1LearningAttackerGuardController(model_path=guard_model_path, greedy=greedy, device=device)

        # 💡追加: チーム内で「サイト内で誰かが既にアビリティを使用したか」を共有する状態。
        # ラウンドごとにreset_roundでクリアする。
        self.ability_advisor = ability_advisor if ability_advisor is not None else FixedAbilityAdvisor()
        self._advisor_scenarios = {name: scenario(name) for name in MAPS}
        digest = policy_digest(paths)
        for context in self._advisor_scenarios.values():
            self.ability_advisor.bind_route(context, digest)
        self.carry_controller.lineup_advice = self._lineup_advice
        self.escort_controller.lineup_advice = self._lineup_advice
        self._advice_cache = {}

        self.site_ability_used_by_team = False

    def _build_or_fallback(self, model_path, param_name):
        if model_path is not None:
            raise NotImplementedError(f"{param_name} に対応するモデルは未実装です。Noneのままにしてください。")
        return DefaultAttackerController()

    def set_game(self, game):
        """run_game.py init_round()より前に一度だけ呼ばれる。
        自分自身のgame参照に加え、各サブコントローラーにも転送する。
        これが無いとcarry/escort/retrieve/guardの各コントローラーは
        self.game=Noneのままとなり、gridアクセスやtarget_plant_pos参照が
        すべて機能しない(サイト選択の上書きも黙ってスキップされる)。"""
        self.game = game
        for ctrl in (self.carry_controller, self.escort_controller, self.retrieve_controller, self.guard_controller):
            if hasattr(ctrl, "set_game"):
                ctrl.set_game(game)

    def reset_round(self):
        for ctrl in (self.carry_controller, self.escort_controller, self.retrieve_controller, self.guard_controller):
            if hasattr(ctrl, "reset_round"):
                ctrl.reset_round()
        self.site_ability_used_by_team = False  # 💡追加

        site = 'L' if self.carry_controller._active_site == 'left' else 'R'
        self.ability_advisor.start_round('carry_' + site,
            getattr(getattr(self, 'game', None), 'current_round', None))

    def decide_move(self, char, game_state):
        game = getattr(self, "game", None)
        self._advice_cache = {}
        self._advice_char, self._advice_state = char, game_state
        if hasattr(self, 'ability_advisor') and not game_state.get('defender_setup_active'):
            self.ability_advisor.observe(self._advisor_context(char), char, game_state,
                                         defending_opponent(game))
        shared = tactical_state(game_state)
        taps = shared.get("defender_defuse_info") or {}
        if game_state.get("is_planted") and any(
                taps.get(enemy.name, (0, 0))[0] > 0 and shot_clear(char, enemy, shared)
                for enemy in known_enemies(char, shared)):
            # Interrupt an immediately shootable defuser before a zone escape
            # can move the shooter out of position during the six-tick window.
            return apply_shared_facing(char, game_state, (list(char.pos), {"facing": char.facing}))
        escape = escape_destruction_area(char, game_state, game)
        if escape is not None:
            return apply_shared_facing(char, game_state, escape)
        result = self._decide_phase_move(char, game_state)
        result = avoid_destruction_entry(char, game_state, result, game)
        if hasattr(self, 'ability_advisor') and self._advice_cache:
            # Candidate scoring is provisional. Track only the target actually returned.
            self.ability_advisor.pending.pop(char.name, None)
            payload = result[1]
            if isinstance(payload, dict) and 'ability' in payload and 'target' in payload:
                key = (payload['ability'], tuple(payload['target']))
                if self._advice_cache.get(key):
                    self.ability_advisor.pending[char.name] = (
                        key[0], key[1], getattr(char, key[0].lower()+'_charges', 0),
                        int(game_state.get('battle_tick', 0)))
        return apply_shared_facing(char, game_state, result)

    def _advisor_context(self, char):
        role = 'carry' if char.has_spike else 'escort'
        site = 'L' if self.carry_controller._active_site == 'left' else 'R'
        return self._advisor_scenarios[role + '_' + site]

    def _lineup_advice(self, ability, point):
        key = (ability, tuple(point))
        if key not in self._advice_cache:
            char, state = self._advice_char, self._advice_state
            self._advice_cache[key] = self.ability_advisor.advise(
                self._advisor_context(char), char, state,
                defending_opponent(getattr(self, 'game', None)), ability,
                point, point, getattr(self, 'game', None)).use
        return self._advice_cache[key]

    def _decide_phase_move(self, char, game_state):
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
