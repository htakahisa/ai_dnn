"""GC v2 attacker; v1 checkpoints stay read-only and remain independently usable."""
import copy
import os
from ghost_champions_v1_macro import GhostChampionsV1AttackerController
from .config import load_config, validate_config
from .tactics import AttackerTactics


class GhostChampionsV2AttackerController(GhostChampionsV1AttackerController):
    requires_public_effects = True
    def __init__(self, greedy=True, config=None, config_path=None, stage=None):
        self.config = copy.deepcopy(config) if config is not None else load_config(config_path)
        validate_config(self.config)
        self.tactics = AttackerTactics(self.config, stage or os.environ.get("GC_V2_STAGE", "entry"))
        super().__init__(greedy=greedy)
        # v2 must not append its diagnostics to v1's historical root log.
        if getattr(self, "guard", None) is not None:
            self.guard.verbose = False

    def reset_round(self):
        super().reset_round()
        self.tactics.reset_round()

    def set_game(self, game):
        # Disable the inherited roster-based A restriction for v2 only.
        owner=getattr(game,"real_game",game)
        owner.gc_opponent_site_overrides=False
        super().set_game(game)

    @property
    def observed_defenders_by_axis(self):
        return dict(self.tactics.observed_defenders_by_axis)

    def decide_move(self, char, game_state):
        settings, flags = self.tactics.observe(char,game_state)
        result = self.tactics.dodge(char,game_state) if char.is_alive else None
        if result is not None:
            self.tactics.remember_result(char,result)
            return result
        if char.is_alive and game_state.get("is_planted") and flags["post_plant_hold"]:
            result = self.tactics.postplant(char,game_state)
        elif char.is_alive and not game_state.get("is_planted") and flags["site_selection"]:
            result = self.tactics.preplant(char,game_state,settings,flags)
        if result is None:
            result = super().decide_move(char,game_state)
            if (flags["early_recon"] and not game_state.get("is_planted")
                    and not (isinstance(result,tuple) and len(result)>1 and result[1] == "PLANT")):
                result = self.tactics.recon(char,game_state) or result
            if flags["entry_discipline"]:
                result = self.tactics.discipline(char,game_state,result)
        result = self.tactics.avoid_entry(char,game_state,result)
        self.tactics.remember_result(char,result)
        return result

    def attacker_snapshot(self):
        return self.tactics.snapshot()
