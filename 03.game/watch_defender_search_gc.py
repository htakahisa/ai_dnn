from __future__ import annotations

import argparse

from controllers import DefaultDefenderController
from map_data import NEW_MAZE_STR
from party_presets import get_preset
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI

from gc_v1.learning_defender_search_gc import LearningDefenderSearchGCController

try:
    from gc_v1.learning_defender_setup_gc_runtime import LearningDefenderSetupGCRuntime
except Exception:
    LearningDefenderSetupGCRuntime = None

try:
    from gc_v1.character_stats_gc import GC_ROSTER_ORDER
except Exception:
    from character_stats_gc import GC_ROSTER_ORDER


class SearchWatchDefenderController:
    """Setup -> Search v3 -> Default Retake.

    Opening is intentionally disabled so Search can be inspected clearly.
    """

    def __init__(self, model_path=None):
        kwargs = {"greedy": True, "verbose": True}
        if model_path:
            kwargs["model_path"] = model_path
        self.search = LearningDefenderSearchGCController(**kwargs)
        self.setup = (
            LearningDefenderSetupGCRuntime(device="cpu", verbose=False)
            if LearningDefenderSetupGCRuntime is not None
            else None
        )
        self.retake = DefaultDefenderController()
        self.game = None

    def set_game(self, game):
        self.game = game
        if hasattr(self.search, "set_game"):
            self.search.set_game(game)
        if self.setup is not None and hasattr(self.setup, "set_game"):
            self.setup.set_game(game)
        if hasattr(self.retake, "set_game"):
            self.retake.set_game(game)

    def reset_round(self):
        if hasattr(self.search, "reset_round"):
            self.search.reset_round()
        if self.setup is not None and hasattr(self.setup, "reset_round"):
            self.setup.reset_round()
        if hasattr(self.retake, "reset_round"):
            self.retake.reset_round()

    def _in_setup(self):
        phase = getattr(self.game, "defender_setup_phase", None)
        return bool(phase is not None and phase.active)

    def decide_move(self, char, game_state):
        if self._in_setup():
            if self.setup is None:
                return list(char.pos)
            chars = game_state.get("chars")
            if chars is None:
                chars = getattr(self.game, "chars", [])
            if not getattr(self.setup, "round_initialized", False):
                self.setup.initialize_round(chars)
            return self.setup.decide_setup_move(char, chars)

        if bool(game_state.get("is_planted", False)):
            return self.retake.decide_move(char, game_state)

        return self.search.decide_move(char, game_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", required=True, help="Exact party preset name")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--attacker-ai", default="toru_ai_v3.1")
    args = ap.parse_args()

    preset = get_preset(args.opponent)
    if preset is None:
        raise ValueError(f"Unknown party preset: {args.opponent!r}")

    defender = SearchWatchDefenderController(args.checkpoint)

    defender_ai = DualRoleTeamAI(
        "GC Search v3 Watch",
        attacker_factory=lambda: DefaultDefenderController(),
        defender_factory=lambda: defender,
        use_iq_perception=False,
    )
    attacker_ai = _build_team_ai(args.attacker_ai)

    game = VisualFPSBattle(
        NEW_MAZE_STR,
        attacker_ai,
        defender_ai,
        headless=False,
        attacker_roster=list(preset.players),
        defender_roster=list(GC_ROSTER_ORDER),
        spike_holder_name=preset.spike_holder,
        attacker_igl_name=preset.igl,
        attacker_team_name=preset.name,
        defender_team_name="Ghost Champions",
        disable_side_swap=True,
    )

    print("=" * 72)
    print("GC SEARCH v3 VISUAL WATCH")
    print("Opening disabled intentionally.")
    print("Console [SEARCH] lines show FORCE_POS / SIGHTING / SPIKE state.")
    print("=" * 72)
    game.run()


if __name__ == "__main__":
    main()
