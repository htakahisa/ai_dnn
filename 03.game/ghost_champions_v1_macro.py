"""Ghost Champions v1 + Attacker Macro + Defender Opening Macro.

Place this file in 03.game root as ghost_champions_v1_macro.py
(or copy its contents into the existing wrapper).

Hierarchy:

Attacker:
    Attacker Macro
        -> Carry / Escort / Retrieve / Guard

Defender:
    Defender Opening Macro
        -> Search (pre-plant)
        -> Retake remains owned by base controller after plant

If the Defender Opening checkpoint does not exist yet, only that layer is
disabled and the existing Ghost Champions Defender behavior remains unchanged.
"""

from __future__ import annotations

from ghost_champions_v1 import (
    GhostChampionsV1AttackerController as _BaseGCAttacker,
    GhostChampionsV1DefenderController as _BaseGCDefender,
)

from learning_attacker_macro_gc_runtime import (
    LearningAttackerMacroGCController,
)
from gc_v1.positioning_gc import team_plant_target

_opening_import_error = None
try:
    from learning_defender_opening_macro_gc_runtime import (
        LearningDefenderOpeningMacroGCRuntime,
    )
except Exception as exc:
    LearningDefenderOpeningMacroGCRuntime = None
    _opening_import_error = f"{type(exc).__name__}: {exc}"

_setup_import_error = None
try:
    from gc_v1.learning_defender_setup_gc_runtime import (
        LearningDefenderSetupGCRuntime,
    )
except Exception as exc:
    LearningDefenderSetupGCRuntime = None
    # except-as の変数はブロック後に消えるため、文字列として保持する。
    _setup_import_error = f"{type(exc).__name__}: {exc}"


class GhostChampionsV1AttackerController(_BaseGCAttacker):
    """Existing Carry/Escort/Retrieve/Guard + Attacker Macro coordinator."""

    def __init__(self, greedy=True):
        super().__init__(greedy=greedy)
        try:
            self.macro_controller = LearningAttackerMacroGCController(
                greedy=True,
                verbose=True,
            )
        except Exception as exc:
            self.macro_controller = None
            print(f"[GC Macro][WARN] attacker macro disabled: {exc}")
        if self.macro_controller is not None:
            self.macro_controller.learned_positioning = (
                getattr(self.carry, "positioning_version", 0) >= 1)

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

    def _cover_result(self, char, game_state, result):
        if getattr(getattr(self, "escort", None), "positioning_version", 0) >= 2:
            return result
        # The team macro owns pre-plant roles. Generic escort nudges otherwise
        # recall fake sellers / split support and destroy the selected plan.
        if (getattr(self, "macro_controller", None) is not None
                and not game_state.get("is_planted")
                and any(getattr(c, "team", None) == "A"
                        and getattr(c, "is_alive", True)
                        and getattr(c, "has_spike", False)
                        for c in game_state.get("chars", []))):
            return result
        return super()._cover_result(char, game_state, result)

    def decide_move(self, char, game_state):
        learned_carry = getattr(getattr(self, "carry", None), "positioning_version", 0) >= 1
        if learned_carry and not game_state.get("is_planted"):
            game_state = dict(game_state)
            target = team_plant_target(getattr(self, "game", None))
            if target is not None:
                game_state["target_plant_pos"] = target
        if (self.macro_controller is not None
                and learned_carry
                and not game_state.get("is_planted")
                and any(getattr(c, "has_spike", False) and c.is_alive
                        for c in game_state.get("chars", []))):
            self.macro_controller._sync_tick_once(game_state)
            target = team_plant_target(getattr(self, "game", None))
            if target is not None:
                game_state["target_plant_pos"] = target
        # Existing phase router first decides Carry/Escort/Retrieve/Guard.
        base_result = super().decide_move(char, game_state)

        if self.macro_controller is None:
            return base_result

        # Guard owns post-plant.
        if bool(game_state.get("is_planted", False)):
            if getattr(getattr(self, "guard", None), "positioning_version", 0) >= 1:
                return base_result
            micro_cover = self._micro_cover_result(char, game_state)
            if micro_cover is not None:
                return micro_cover
            return base_result

        # Retrieve owns dropped-spike phase.
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

        # Macro selects the target site; retrained Carry owns its route and
        # planting. Its training includes the complete spawn-to-site approach.
        if (char is holder and getattr(self.carry, "positioning_version", 0) >= 1):
            return base_result

        # Intent-trained Escort executes its own actions. Macro has already
        # published its role/waypoint at _sync_tick_once above; overriding this
        # result would make the recorded DQN action differ from its execution.
        if char is not holder and getattr(self.escort, "positioning_version", 0) >= 2:
            return base_result

        coordinated = self.macro_controller.coordinate(
            char,
            game_state,
            base_result,
        )
        return coordinated


class GhostChampionsV1DefenderController(_BaseGCDefender):
    """Setup -> Opening -> existing Search/Retake hierarchy."""

    def __init__(self, greedy=True):
        super().__init__(greedy=greedy)

        self.game = None
        self.setup_planner = None
        self.opening_macro_controller = None

        # --------------------------------------------------------------
        # Defender Setup layer
        # --------------------------------------------------------------
        if LearningDefenderSetupGCRuntime is None:
            print(
                "[GC D-SETUP][WARN] setup planner import failed; "
                f"disabled: {_setup_import_error}"
            )
        else:
            try:
                self.setup_planner = LearningDefenderSetupGCRuntime(
                    verbose=True,
                )
            except Exception as exc:
                self.setup_planner = None
                print(
                    "[GC D-SETUP][WARN] disabled: "
                    f"{exc}"
                )

        # --------------------------------------------------------------
        # Defender Opening layer
        # --------------------------------------------------------------
        if LearningDefenderOpeningMacroGCRuntime is None:
            print(
                "[GC D-OPENING][WARN] runtime import failed; "
                f"disabled: {_opening_import_error}"
            )
        else:
            try:
                self.opening_macro_controller = (
                    LearningDefenderOpeningMacroGCRuntime(
                        greedy=True,
                        verbose=True,
                    )
                )
            except FileNotFoundError:
                print(
                    "[GC D-OPENING] no trained checkpoint yet; "
                    "Opening Macro disabled"
                )
            except Exception as exc:
                print(
                    "[GC D-OPENING][WARN] disabled: "
                    f"{exc}"
                )

    def set_game(self, game):
        parent = getattr(super(), "set_game", None)
        if callable(parent):
            parent(game)
        else:
            self.game = game

        self.game = game

        if self.setup_planner is not None and hasattr(
            self.setup_planner,
            "set_game",
        ):
            self.setup_planner.set_game(game)

        if self.opening_macro_controller is not None:
            self.opening_macro_controller.set_game(game)

    def reset_round(self):
        super().reset_round()

        if self.setup_planner is not None:
            self.setup_planner.reset_round()

        if self.opening_macro_controller is not None:
            self.opening_macro_controller.reset_round()

    def _in_defender_setup_phase(self):
        phase = getattr(self.game, "defender_setup_phase", None)
        return bool(phase is not None and phase.active)

    def _setup_move(self, char, game_state):
        if self.setup_planner is None:
            return None

        chars = game_state.get("chars")
        if chars is None:
            chars = getattr(self.game, "chars", [])

        if not self.setup_planner.round_initialized:
            assignments = self.setup_planner.initialize_round(chars)
            print(
                "[GC D-SETUP] assignments="
                + ", ".join(
                    f"{a.player_name}->{a.target}"
                    for a in assignments
                )
            )

        return self.setup_planner.decide_setup_move(
            char,
            chars,
        )

    def decide_move(self, char, game_state):
        # --------------------------------------------------------------
        # Layer 1: Defender Setup
        # --------------------------------------------------------------
        # During Setup, do NOT call base Search/Retake or Opening first.
        # Setup owns movement completely and abilities are already disabled
        # by the global Setup Phase.
        if self._in_defender_setup_phase():
            setup_result = self._setup_move(char, game_state)
            if setup_result is not None:
                return setup_result

            # Fallback if planner is unavailable.
            return list(char.pos)

        # --------------------------------------------------------------
        # Layer 2+: Existing GC Search/Retake
        # --------------------------------------------------------------
        base_result = super().decide_move(char, game_state)

        if self.opening_macro_controller is None:
            return base_result

        # Retake remains untouched.
        if bool(game_state.get("is_planted", False)):
            return base_result

        # --------------------------------------------------------------
        # Layer 2: Defender Opening Macro
        # --------------------------------------------------------------
        return self.opening_macro_controller.coordinate(
            char,
            game_state,
            base_result,
        )

    def setup_snapshot(self):
        if self.setup_planner is None:
            return None

        chars = getattr(self.game, "chars", None)
        return self.setup_planner.snapshot(chars)


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
