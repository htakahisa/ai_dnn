"""Integration checks for the per-team controller and actor information boundary."""

import unittest

import numpy as np

from map_data import NEW_MAZE_STR
from team_ai import DualRoleTeamAI

from coach_v1.common.types import Facing, MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.coordinator import TeamExecutionCoordinator
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.training.character_environment import CharacterAction
from coach_v1.common.constants import FIXED_ROSTER
from coach_v1.observation.coach_encoder import COACH_GRID_CHANNELS


class DummyCoach:
    def __init__(self, movement=MovementAction.MOVE_E):
        self.calls = []
        self.movement = movement

    def act(self, observation):
        self.calls.append((observation.grid.copy(), observation.vector.copy()))
        return tuple(CoachInstruction(self.movement, ObjectiveAction.NONE,
                                      TacticalIntent.ADVANCE) for _ in range(5))


class DummyCharacter:
    def __init__(self, action=None):
        self.action = action or CharacterAction(Facing.N)
        self.calls = []

    def act(self, observation):
        self.calls.append((observation.grid.copy(), observation.vector.copy()))
        return self.action


def make_game(side=Side.ATTACKER, enemy_pos=(4, 6)):
    grid = np.array([[int(cell) for cell in line]
                     for line in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
    code = "A" if side is Side.ATTACKER else "D"
    other = "D" if code == "A" else "A"
    allies = [FakeCharacter(slot.character_name, code, (23, 18 + i),
                            facing="E", alive=True)
              for i, slot in enumerate(FIXED_ROSTER)]
    enemy = FakeCharacter("enemy", other, enemy_pos, facing="N")
    return FakeGame(grid, allies + [enemy]), allies, enemy


def make_coordinator(game, side=Side.ATTACKER, *, coach=None, actors=None):
    coach = coach or DummyCoach()
    actors = actors or {i: DummyCharacter() for i in range(5)}
    controller = TeamExecutionCoordinator(side, coach, actors)
    controller.set_game(game)
    return controller, coach, actors


class CoordinatorTest(unittest.TestCase):
    def test_one_coach_call_and_snapshot_stays_fixed_after_first_move(self):
        game, allies, _ = make_game()
        controller, coach, actors = make_coordinator(game)
        first = controller.decide_move(allies[0], {})
        self.assertEqual((23, 19), first[0])
        allies[0].pos = [22, 18]
        for ally in allies[1:]:
            controller.decide_move(ally, {})
        self.assertEqual(1, len(coach.calls))
        self.assertEqual(1, len(actors[0].calls))
        self.assertEqual(5, len(controller.action_log))
        self.assertEqual((23, 18), controller._snapshot.allies[0].position)
        self.assertEqual((23, 18), controller.action_log[0].start)

    def test_first_character_order_does_not_change_coach_input(self):
        game_a, allies_a, _ = make_game()
        game_b, allies_b, _ = make_game()
        controller_a, coach_a, _ = make_coordinator(game_a)
        controller_b, coach_b, _ = make_coordinator(game_b)
        controller_a.decide_move(allies_a[0], {})
        controller_b.decide_move(allies_b[4], {})
        np.testing.assert_array_equal(coach_a.calls[0][0], coach_b.calls[0][0])
        np.testing.assert_array_equal(coach_a.calls[0][1], coach_b.calls[0][1])

    def test_ability_stays_put_and_applies_facing(self):
        game, allies, _ = make_game()
        start = tuple(allies[0].pos)
        actors = {i: DummyCharacter() for i in range(5)}
        actors[0] = DummyCharacter(CharacterAction(Facing.SW, True, start))
        controller, _, _ = make_coordinator(game, actors=actors)
        result = controller.decide_move(allies[0], {})
        self.assertEqual(start, result[0])
        self.assertEqual("SMOKE", result[1]["ability"])
        self.assertEqual("SW", allies[0].facing)
        self.assertEqual("ABILITY", controller.action_log[0].action)
        self.assertEqual(start, controller.action_log[0].requested_position)

    def test_character_cannot_override_coach_movement(self):
        game, allies, _ = make_game()
        controller, _, _ = make_coordinator(game, coach=DummyCoach(MovementAction.MOVE_W))
        result = controller.decide_move(allies[0], {})
        self.assertEqual((23, 17), result[0])

    def test_round_and_setup_transition_clear_tick_cache_and_dead_slot_skips_actor(self):
        game, allies, _ = make_game()
        controller, coach, actors = make_coordinator(game)
        game.defender_setup_phase.active = True
        game.defender_setup_phase.ticks_remaining = 3
        controller.decide_move(allies[0], {})
        game.defender_setup_phase.ticks_remaining = 2
        controller.decide_move(allies[1], {})
        self.assertEqual(2, len(coach.calls))
        game.defender_setup_phase.active = False
        controller.decide_move(allies[2], {})
        self.assertEqual(3, len(coach.calls))
        game.current_round += 1
        allies[3].is_alive = False
        result = controller.decide_move(allies[3], {})
        self.assertEqual(tuple(allies[3].pos), result[0])
        self.assertEqual(4, len(coach.calls))
        self.assertEqual(0, len(actors[3].calls))
        self.assertEqual(0, controller._belief.memory_tick)

    def test_unseen_enemy_location_cannot_change_actor_observation(self):
        game_a, allies_a, enemy_a = make_game(enemy_pos=(4, 6))
        game_b, allies_b, enemy_b = make_game(enemy_pos=(4, 7))
        controller_a, coach_a, actors_a = make_coordinator(game_a)
        controller_b, coach_b, actors_b = make_coordinator(game_b)
        controller_a.decide_move(allies_a[0], {})
        controller_b.decide_move(allies_b[0], {})
        self.assertNotEqual(tuple(enemy_a.pos), tuple(enemy_b.pos))
        self.assertFalse(controller_a._snapshot.sightings)
        self.assertFalse(controller_b._snapshot.sightings)
        np.testing.assert_array_equal(coach_a.calls[0][0], coach_b.calls[0][0])
        np.testing.assert_array_equal(coach_a.calls[0][1], coach_b.calls[0][1])
        np.testing.assert_array_equal(actors_a[0].calls[0][0], actors_b[0].calls[0][0])
        np.testing.assert_array_equal(actors_a[0].calls[0][1], actors_b[0].calls[0][1])

    def test_legal_sighting_is_shared_and_memory_ages(self):
        game, allies, _ = make_game(enemy_pos=(22, 22))
        controller, coach, actors = make_coordinator(game)
        controller.decide_move(allies[0], {})
        self.assertEqual(1, len(controller._snapshot.sightings))
        sighting = controller._snapshot.sightings[0]
        row, column = sighting.reported_position
        channel = COACH_GRID_CHANNELS.index("current_enemy_sighting")
        self.assertGreater(coach.calls[0][0][channel, row, column], 0)
        controller.decide_move(allies[4], {})
        np.testing.assert_array_equal(
            actors[0].calls[0][0][:27], actors[4].calls[0][0][:27],
        )
        for ally in allies:
            ally.blind_remaining = 1
        game.battle_tick += 1
        controller.decide_move(allies[0], {})
        self.assertFalse(controller._snapshot.sightings)
        self.assertEqual(1, controller._belief.last_seen_age[row][column])
        self.assertEqual(1, controller._belief.clear_age[row][column])

    def test_rejects_other_map(self):
        game, _, _ = make_game()
        game.grid[22, 22] = 1
        with self.assertRaisesRegex(ValueError, "fixed map"):
            make_coordinator(game)

    def test_team_ai_uses_native_team_perception_controller(self):
        game, _, _ = make_game()
        controller, _, _ = make_coordinator(game)
        team = DualRoleTeamAI("coach", lambda: controller, lambda: controller)
        team.bind_game(game)
        self.assertIs(controller, team.get_attacker_controller())


if __name__ == "__main__":
    unittest.main()
