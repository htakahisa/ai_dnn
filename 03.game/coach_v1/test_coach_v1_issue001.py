"""Regression checks for Issue 001 training and positional inputs."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from map_data import NEW_MAZE_STR
from coach_v1.models.coach_model import CoachActorModel, CoachModelConfig
from coach_v1.observation.coach_encoder import COACH_GRID_CHANNELS, COACH_VECTOR_FIELDS
from coach_v1.training.coach_environment import _balanced_plant_cell
from coach_v1.training.coach_environment import CoachTrainingEnvironment
from coach_v1.training.defender_teacher import _distance_map
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.defender_imitation import fit_balanced_imitation
from coach_v1.common.types import Side


class Issue001Tests(unittest.TestCase):
    def test_training_samples_physical_sites_equally(self):
        sites = [(r, c) for r, row in enumerate(NEW_MAZE_STR.strip().splitlines())
                 for c, tile in enumerate(row) if tile == "2"]
        selected = [_balanced_plant_cell(sites, index) for index in range(42)]
        self.assertEqual(21, sum(column < 22 for _, column in selected))
        self.assertEqual(21, sum(column >= 22 for _, column in selected))
        self.assertEqual(set(sites), set(selected))

    def test_actor_receives_distinct_left_and_right_spike_geometry(self):
        torch.manual_seed(41)
        model = CoachActorModel(CoachModelConfig(
            spatial_coordinates=True, objective_geometry=True,
            local_spatial=True)).eval()
        base = torch.zeros(1, len(COACH_GRID_CHANNELS), 26, 44)
        base[0, COACH_GRID_CHANNELS.index("plantable"), 9, 3] = 1
        base[0, COACH_GRID_CHANNELS.index("plantable"), 8, 40] = 1
        vector = torch.zeros(1, len(COACH_VECTOR_FIELDS))
        left, right = base.clone(), base.clone()
        left[0, COACH_GRID_CHANNELS.index("spike_planted"), 9, 3] = 1
        right[0, COACH_GRID_CHANNELS.index("spike_planted"), 8, 40] = 1
        with torch.no_grad():
            left_move = model(left, vector)[0]
            right_move = model(right, vector)[0]
        self.assertFalse(torch.allclose(left_move, right_move))

    def test_paired_retake_batch_contains_both_sites(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as folder:
            trainer = CoachTrainer(Side.DEFENDER, seed=11, directory=Path(folder))
            history = fit_balanced_imitation(
                trainer, cycles=1, stage_ticks={"group_up": 1},
                samples_per_bucket=1, epochs=1, paired_sites=True,
            )
        counts = history[-1]["bucket_counts"]
        self.assertGreater(counts["group_up_left"], 0)
        self.assertGreater(counts["group_up_right"], 0)

    def test_long_distance_retake_curriculum_reaches_realistic_range(self):
        environment = CoachTrainingEnvironment(
            Side.DEFENDER, seed=12, stage="group_up", max_ticks=45)
        environment.reset()
        planted = tuple(environment.game.planted_pos)
        distances = _distance_map(planted)
        allies = [char for char in environment.game.chars if char.team == "D"]
        self.assertGreaterEqual(max(distances[tuple(char.pos)] for char in allies), 40)


if __name__ == "__main__":
    unittest.main()
