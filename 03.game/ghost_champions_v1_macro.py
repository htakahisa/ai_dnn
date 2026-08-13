"""
Ghost Champions v1 + Macro DQN practical integration.

Use this module in run_game.py instead of importing the attacker controller
directly from ghost_champions_v1.py. Defender remains unchanged.
"""

from __future__ import annotations

from ghost_champions_v1 import (
    GhostChampionsV1AttackerController as _BaseGCAttacker,
    GhostChampionsV1DefenderController,
)
from learning_attacker_macro_gc_runtime import (
    LearningAttackerMacroGCController,
)


class GhostChampionsV1AttackerController(_BaseGCAttacker):
    """Existing Carry/Escort/Retrieve/Guard + v28 team Macro coordinator."""

    def __init__(self, greedy=True):
        super().__init__(greedy=greedy)
        try:
            self.macro_controller = LearningAttackerMacroGCController(
                greedy=True,
                verbose=True,
            )
        except Exception as exc:
            self.macro_controller = None
            print(f"[GC Macro][WARN] disabled: {exc}")

    def set_game(self, game):
        parent = getattr(super(), "set_game", None)
        if callable(parent):
            parent(game)
        else:
            self.game = game

        if self.macro_controller is not None:
            self.macro_controller.set_game(game)

    def reset_round(self):
        super().reset_round()
        if self.macro_controller is not None:
            self.macro_controller.reset_round()

    def decide_move(self, char, game_state):
        # Existing router keeps ownership of Guard and Retrieve.
        base_result = super().decide_move(char, game_state)

        if self.macro_controller is None:
            return base_result

        if bool(game_state.get("is_planted", False)):
            return base_result

        holder = next(
            (
                c for c in game_state.get("chars", [])
                if getattr(c, "team", None) == "A"
                and bool(getattr(c, "is_alive", True))
                and bool(getattr(c, "has_spike", False))
            ),
            None,
        )
        if holder is None:
            return base_result

        return self.macro_controller.coordinate(
            char,
            game_state,
            base_result,
        )


def build_ghost_champions_v1_team_ai():
    from team_ai import DualRoleTeamAI

    return DualRoleTeamAI(
        name="Ghost Champions v1",
        attacker_factory=lambda: GhostChampionsV1AttackerController(
            greedy=True
        ),
        defender_factory=lambda: GhostChampionsV1DefenderController(
            greedy=True
        ),
    )
