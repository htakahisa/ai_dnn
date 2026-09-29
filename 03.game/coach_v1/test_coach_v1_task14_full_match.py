"""Task 14 full-match runtime, lifecycle, replay, and privacy checks."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controllers import DefaultAttackerController, DefaultDefenderController
from defender_setup_phase import DefenderSetupPhase
from party_presets import get_preset
from team_ai import DualRoleTeamAI

from coach_v1.common.constants import CHARACTER_CHECKPOINT_IDS, FIXED_ROSTER
from coach_v1.common.types import (
    Facing,
    MovementAction,
    ObjectiveAction,
    Side,
    TacticalIntent,
)
from coach_v1.coordinator import TeamExecutionCoordinator
from coach_v1.full_match import run_headless_full_match, write_match_artifacts
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.perception.team_perception import TeamPerceptionBuilder
from coach_v1.team_ai import build_coach_v1_team
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.training.character_environment import CharacterAction
from map_data import NEW_MAZE_STR


class StayCoach:
    def __init__(self, label="stay"):
        self.label = label
        self.calls = []

    def reset_round(self):
        return None

    def act(self, observation):
        self.calls.append((observation.grid.copy(), observation.vector.copy()))
        return tuple(
            CoachInstruction(
                MovementAction.STAY,
                ObjectiveAction.NONE,
                TacticalIntent.HOLD,
            )
            for _ in FIXED_ROSTER
        )


class FacingActor:
    def __init__(self, path=None, *, device="cpu"):
        self.path = path
        self.device = device
        self.calls = []

    def act(self, observation):
        self.calls.append((observation.grid.copy(), observation.vector.copy()))
        return CharacterAction(Facing.N)


def fixed_game(enemy_position):
    grid = np.array(
        [[int(cell) for cell in line]
         for line in NEW_MAZE_STR.strip().splitlines()],
        dtype=np.int8,
    )
    allies = [
        FakeCharacter(slot.character_name, "A", (23, 18 + slot.slot), facing="E")
        for slot in FIXED_ROSTER
    ]
    enemy = FakeCharacter("hidden", "D", enemy_position, facing="N")
    return FakeGame(grid, allies + [enemy]), allies


def stay_team():
    def coordinator(side):
        return TeamExecutionCoordinator(
            side,
            StayCoach(side.value),
            {slot: FacingActor() for slot in range(5)},
        )

    return DualRoleTeamAI(
        "coach_v1_test",
        lambda: coordinator(Side.ATTACKER),
        lambda: coordinator(Side.DEFENDER),
        use_iq_perception=False,
    )


class TeamFactoryTest(unittest.TestCase):
    def test_factory_loads_distinct_side_checkpoints_and_all_character_slots(self):
        loaded = []

        def attacker(path=None, *, device="cpu"):
            loaded.append(("attacker", path, device))
            return StayCoach("attacker")

        def defender(path=None, *, device="cpu"):
            loaded.append(("defender", path, device))
            return StayCoach("defender")

        attacker_path = Path("attacker.pt")
        defender_path = Path("defender.pt")
        character_paths = {
            checkpoint_id: Path(f"{checkpoint_id}.pt")
            for checkpoint_id in CHARACTER_CHECKPOINT_IDS
        }
        with (
            patch("coach_v1.team_ai.load_attacker_coach", attacker),
            patch("coach_v1.team_ai.load_defender_coach", defender),
            patch("coach_v1.team_ai._CHARACTER_POLICY_TYPES", (FacingActor,) * 5),
        ):
            team = build_coach_v1_team(
                attacker_checkpoint=attacker_path,
                defender_checkpoint=defender_path,
                character_checkpoints=character_paths,
                device="cpu",
            )
            attacker_controller = team.get_attacker_controller()
            defender_controller = team.get_defender_controller()

        self.assertIs(Side.ATTACKER, attacker_controller.side)
        self.assertIs(Side.DEFENDER, defender_controller.side)
        self.assertEqual(
            [("attacker", attacker_path, "cpu"),
             ("defender", defender_path, "cpu")],
            loaded,
        )
        for controller in (attacker_controller, defender_controller):
            self.assertEqual(set(range(5)), set(controller.characters))
            self.assertEqual(
                character_paths,
                {
                    CHARACTER_CHECKPOINT_IDS[slot]: actor.path
                    for slot, actor in controller.characters.items()
                },
            )

    def test_factory_rejects_unknown_character_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "unknown character"):
            with patch("coach_v1.team_ai.load_attacker_coach", return_value=StayCoach()):
                team = build_coach_v1_team(
                    character_checkpoints={"not-a-slot": Path("bad.pt")}
                )
                team.get_attacker_controller()

    def test_end_to_end_factory_does_not_expose_unseen_enemy_position(self):
        observations = []
        for enemy_position in ((4, 6), (4, 7)):
            with (
                patch("coach_v1.team_ai.load_attacker_coach",
                      return_value=StayCoach("attacker")),
                patch("coach_v1.team_ai._CHARACTER_POLICY_TYPES", (FacingActor,) * 5),
            ):
                team = build_coach_v1_team()
                game, allies = fixed_game(enemy_position)
                team.bind_game(game)
                controller = team.get_attacker_controller()
                controller.decide_move(allies[0], {})
                self.assertFalse(controller._snapshot.sightings)
                observations.append((
                    controller.coach.calls[0],
                    controller.characters[0].calls[0],
                ))
        for left, right in zip(observations[0], observations[1]):
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(left[1], right[1])


class FullMatchTest(unittest.TestCase):
    def test_real_headless_match_completes_swaps_sides_resets_memory_and_replays(self):
        preset = get_preset("Ghost Champions")
        opponent = DualRoleTeamAI(
            "default_test",
            DefaultAttackerController,
            DefaultDefenderController,
        )
        def early_halftime(game):
            if not game.sides_swapped and game.current_round >= 2:
                game._swap_sides()
                game.sides_swapped = True

        # Short timers and an early halftime exercise the same game-owned
        # lifecycle in four rounds instead of running a production-length map.
        with (
            patch("run_game.ROUND_DURATION_TICKS", 1),
            patch("battle_logic.WINNING_ROUNDS", 3),
            patch("run_game.VisualFPSBattle._swap_sides_if_needed", early_halftime),
            patch("run_game.DefenderSetupPhase",
                  side_effect=lambda: DefenderSetupPhase(setup_ticks=2)),
            # LOS/facing correctness has dedicated Task 03 and privacy tests.
            # This test keeps the real DTO/encoder/coordinator/game lifecycle
            # while avoiding a full-map LOS sweep for every audit tick.
            patch.object(TeamPerceptionBuilder, "_visible_cells",
                         return_value=frozenset()),
            patch.object(TeamPerceptionBuilder, "_sighting_for",
                         return_value=None),
        ):
            result = run_headless_full_match(
                coach_team_ai=stay_team(),
                opponent_team_ai=opponent,
                opponent_roster=preset.players,
                opponent_spike_holder=preset.spike_holder,
                opponent_igl=preset.igl,
                seed=1400,
                opponent_team_name=preset.name,
            )

        summary = result.summary
        self.assertEqual(4, summary.rounds)
        self.assertEqual(4, summary.setup_rounds)
        self.assertEqual(4, summary.live_rounds)
        self.assertEqual(4, summary.timeout_rounds)
        self.assertTrue(summary.memory_reset_ok)
        self.assertTrue(summary.side_checkpoint_switch_ok)
        self.assertTrue(summary.replay_ok)
        self.assertGreater(summary.attacker_decisions, 0)
        self.assertGreater(summary.defender_decisions, 0)
        self.assertGreater(summary.replay_frames, summary.rounds)

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            replay = Path(directory) / "replay.json"
            write_match_artifacts(result, report_path=report, replay_path=replay)
            self.assertTrue(report.is_file())
            self.assertTrue(replay.is_file())
            self.assertIn('"memory_reset_ok": true', report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
