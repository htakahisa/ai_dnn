"""Regression checks for information boundaries reviewed in Task 11.5."""

import ast
from pathlib import Path
import unittest

import numpy as np

from map_data import NEW_MAZE_STR

from coach_v1.common.types import Side
from coach_v1.training.coach_environment import CoachTrainingEnvironment


class Task115AuditTest(unittest.TestCase):
    def test_unseen_enemy_relocation_changes_only_critic_for_both_sides(self):
        rows = NEW_MAZE_STR.strip().splitlines()
        for side in Side:
            with self.subTest(side=side):
                environment = CoachTrainingEnvironment(side, seed=5, stage="2v1", max_ticks=1)
                environment.reset()
                enemy_code = "D" if side is Side.ATTACKER else "A"
                enemy = next(character for character in environment.game.chars
                             if character.team == enemy_code and character.is_alive)
                ally_positions = {tuple(character.pos) for character in environment.game.chars
                                  if character.team != enemy_code and character.is_alive}
                states = []
                for row, cells in enumerate(rows):
                    for column, cell in enumerate(cells):
                        if cell == "1" or (row, column) in ally_positions:
                            continue
                        enemy.pos = [row, column]
                        environment.memory.reset()
                        environment.controller.reset_round()
                        snapshot = environment.sensor.build(game=environment.game, side=side)
                        if snapshot.sightings:
                            continue
                        state = environment._observe()
                        states.append(state)
                        if len(states) == 2:
                            break
                    if len(states) == 2:
                        break
                self.assertEqual(2, len(states), "two legal unseen enemy positions are required")
                np.testing.assert_array_equal(states[0].observation.grid,
                                              states[1].observation.grid)
                np.testing.assert_array_equal(states[0].observation.vector,
                                              states[1].observation.vector)
                self.assertFalse(np.array_equal(states[0].critic_enemy_truth,
                                                states[1].critic_enemy_truth))

    def test_learning_entry_points_do_not_import_other_models_train_modules(self):
        root = Path(__file__).resolve().parent
        for entry in root.glob("learning*.py"):
            with self.subTest(entry=entry.name):
                tree = ast.parse(entry.read_text(encoding="utf-8"))
                modules = [node.module or "" for node in ast.walk(tree)
                           if isinstance(node, ast.ImportFrom)]
                modules += [alias.name for node in ast.walk(tree)
                            if isinstance(node, ast.Import) for alias in node.names]
                self.assertFalse([module for module in modules
                                  if any(part.startswith("train_") for part in module.split("."))])


if __name__ == "__main__":
    unittest.main()
