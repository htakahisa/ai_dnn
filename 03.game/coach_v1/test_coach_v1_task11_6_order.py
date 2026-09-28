"""Real move-order regression tests for the Task 11.6 curriculum."""

import unittest
from pathlib import Path
import tempfile

import numpy as np

from coach_v1.common.types import Facing, MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.training.character_environment import CharacterAction
from coach_v1.training.coach_environment import CoachTrainingEnvironment, _STAY
from coach_v1.training.coach_trainer import CoachTrainer


class _NoAbility:
    def act(self, observation):
        return CharacterAction(Facing.N)


class MoveOrderTest(unittest.TestCase):
    def test_trainer_rollout_stores_actual_actor_input_on_both_sides(self):
        with tempfile.TemporaryDirectory() as root:
            for side in Side:
                with self.subTest(side=side):
                    trainer = CoachTrainer(side, seed=13,
                                           directory=Path(root) / side.value)
                    environment = CoachTrainingEnvironment(side, seed=13,
                                                            max_ticks=2)
                    trajectory, metrics = trainer.collect(environment, episode=0)
                    self.assertEqual(2, metrics["ticks"])
                    self.assertEqual(2, len(trajectory))
                    np.testing.assert_array_equal(trajectory[-1].grid,
                                                  environment.queue.last_observation.grid)
                    np.testing.assert_array_equal(trajectory[-1].vector,
                                                  environment.queue.last_observation.vector)

    def test_rollout_executes_each_character_once_in_game_order(self):
        for side in Side:
            with self.subTest(side=side):
                environment = CoachTrainingEnvironment(side, seed=7, max_ticks=1)
                environment.reset()
                game = environment.game
                own_code = "A" if side is Side.ATTACKER else "D"
                for character in game.chars:
                    character.has_spike = False
                if side is Side.ATTACKER:
                    carrier = [character for character in game.chars
                               if character.team == own_code and character.is_alive][1]
                else:
                    carrier = next(character for character in game.chars
                                   if character.team != own_code and character.is_alive)
                carrier.has_spike = True
                game.spike_pos = None
                expected = tuple(game._move_order())
                self.assertIs(carrier, expected[0])
                first_own = next(i for i, character in enumerate(expected)
                                 if character.team == own_code)
                self.assertEqual(0 if side is Side.ATTACKER else 1, first_own)
                calls = []
                original_move = game.move_character

                def record_move(character):
                    calls.append(character)
                    return original_move(character)

                game.move_character = record_move
                environment.controller.reset_round()
                environment.memory.reset()
                environment._prepare_tick()
                environment.state = environment._observe()
                self.assertEqual(expected[:first_own], tuple(calls))
                self.assertEqual(expected[first_own:], environment._pending_moves)
                before = environment.state.observation
                environment.step(_STAY)
                self.assertEqual(expected, tuple(calls))
                np.testing.assert_array_equal(before.grid, environment.queue.last_observation.grid)
                np.testing.assert_array_equal(before.vector, environment.queue.last_observation.vector)

    def test_defender_observes_enemy_vacating_cell_before_move(self):
        environment = CoachTrainingEnvironment(Side.DEFENDER, seed=7, max_ticks=1)
        environment.reset()
        game = environment.game
        allies = [character for character in game.chars
                  if character.team == "D" and character.is_alive]
        carrier = next(character for character in game.chars
                       if character.team == "A" and character.is_alive)
        for character in game.chars:
            character.has_spike = False
        carrier.has_spike = True
        game.spike_pos = None
        allies[0].pos = [23, 18]
        allies[0].facing = Facing.NW
        allies[1].pos = [23, 19]
        carrier.pos = [23, 17]
        environment.controller.reset_round()
        environment.memory.reset()
        original_move = game.move_character

        def vacate(character):
            if character is carrier:
                old = tuple(character.pos)
                character.pos = [22, 17]
                game._update_occupancy_after_move(old, tuple(character.pos))
                return None
            return original_move(character)

        game.move_character = vacate
        environment._prepare_tick()
        environment.state = environment._observe()
        self.assertEqual((22, 17), tuple(carrier.pos))
        self.assertTrue(environment.sensor.build(game=game, side=Side.DEFENDER).sightings)
        environment.controller.characters = {slot: _NoAbility() for slot in range(5)}
        actions = list(_STAY)
        actions[0] = CoachInstruction(MovementAction.MOVE_W,
                                      ObjectiveAction.NONE, TacticalIntent.HOLD)
        transition = environment.step(tuple(actions))
        self.assertEqual((23, 17), tuple(allies[0].pos))
        self.assertEqual(1.0, transition.metrics["executed_moves"])
        self.assertEqual(0.0, transition.metrics["invalid_moves"])


if __name__ == "__main__":
    unittest.main()
