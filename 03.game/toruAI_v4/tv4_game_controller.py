"""Game integration: opponent-specific learned phases, generic unfinished phases."""
from pathlib import Path
import hashlib
import numpy as np

from controllers import DefaultDefenderController
from iq_controller_adapter import IQAwareController
from toruAI_v4.tv4_scenario import Scenario
from toruAI_v4.tv4_defender_controller import (
    ToruV4DefenderController, OPPONENT_NAMES, DEFENDER_BEST, load_analyses, load_policy,
)


class ToruV4GameDefenderController:
    handles_team_perception = True

    def __init__(self, *, best_dir=None, analysis_dir=None):
        self.best_dir = Path(best_dir) if best_dir else DEFENDER_BEST
        self.analysis_dir = analysis_dir
        self.scenario = Scenario()
        self.generic = IQAwareController(DefaultDefenderController(), viewer_team="D")
        self.game = self.learned = self.opponent = None
        self.binding = None
        self.status = {}

    def set_game(self, game):
        real = getattr(game, "_real", game)
        name = real.current_attacker_team_ai.name
        opponent = OPPONENT_NAMES.get(name)
        binding = (id(real), name)
        self.game = real
        self.generic.set_game(real)
        if binding == self.binding:
            return
        self.binding, self.opponent, self.learned = binding, opponent, None
        self.status = {phase: "汎用controller" for phase in ("search", "retake_L", "retake_R")}
        if opponent is None:
            self._report(name, "相手専用モデル未対応")
            return
        if not np.array_equal(real.grid, self.scenario.grid):
            self._report(name, "学習時と異なるマップ")
            return
        source = self.best_dir / opponent / "search_best.pt"
        try:
            search, saved = load_policy(source, "search", self.scenario)
            if saved.get("opponent") != opponent:
                raise ValueError("searchの相手AIが不一致")
            analyses, signatures = load_analyses(self.scenario, (opponent,), self.analysis_dir)
            if saved.get("analysis_hashes", {}).get(opponent) != signatures[opponent]:
                raise ValueError("解析モデルがsearch学習時と不一致")
        except (FileNotFoundError, ValueError) as error:
            self._report(name, str(error))
            return
        retakes = {}
        search_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        self.status["search"] = f"学習済み: {source}"
        for side in ("L", "R"):
            path = self.best_dir / opponent / f"retake_{side}_best.pt"
            try:
                model, state = load_policy(path, "retake", self.scenario)
                if state.get("opponent") != opponent or state.get("site") != side:
                    raise ValueError("retakeの相手AI・サイトが不一致")
                if state.get("frozen_search_hash") != search_hash or state.get("analysis_hashes", {}).get(opponent) != signatures[opponent]:
                    raise ValueError("retakeの学習元モデルが不一致")
            except (FileNotFoundError, ValueError) as error:
                self.status[f"retake_{side}"] += f" ({error})"
            else:
                retakes[side] = model
                self.status[f"retake_{side}"] = f"学習済み: {path}"
        self.learned = ToruV4DefenderController(opponent, scenario=self.scenario,
            search=search, retake=retakes, analyses=analyses)
        self.learned.set_game(real)
        self._report(name)

    def _report(self, name, reason=None):
        print(f"[Toru AI v4] 相手AI={name}" + (f" 汎用へ切替: {reason}" if reason else ""))
        for phase, status in self.status.items():
            print(f"  {phase}={status}")

    def _uses_learned_phase(self):
        if self.learned is None:
            return False
        if not self.game.is_planted:
            return True
        return self.scenario.site_of(self.game.planted_pos) in self.learned.retake

    def prepare_team_tick(self):
        if self._uses_learned_phase():
            self.learned.prepare_team_tick()

    def decide_move(self, char, state):
        # Recheck after attacker movement: planting can complete mid-tick.
        if self._uses_learned_phase():
            return self.learned.decide_move(char, state)
        return self.generic.decide_move(char, state)

    def reset_round(self):
        self.generic.reset_round()
        if self.learned is not None:
            self.learned.reset_round()

    def record_opponent_round_end(self):
        if self.learned is not None:
            self.learned.record_opponent_round_end()
