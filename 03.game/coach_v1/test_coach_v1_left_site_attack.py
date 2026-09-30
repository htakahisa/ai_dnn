"""Orb-aware observation and left-site candidate compatibility checks."""

import dataclasses
from pathlib import Path
import tempfile
import unittest

import numpy as np

from map_data import NEW_MAZE_STR

from coach_v1.common.types import Side
from coach_v1.common.versions import ORB_COACH_OBSERVATION_VERSION
from coach_v1.left_site_attack import LeftSiteAttackPolicy
from coach_v1.learning_coach_attacker import load_attacker_coach
from coach_v1.models.coach_model import legal_action_mask
from coach_v1.observation.coach_encoder import (
    COACH_GRID_CHANNELS, COACH_VECTOR_FIELDS, ORB_COACH_GRID_CHANNELS,
    ORB_COACH_VECTOR_FIELDS, CoachObservationEncoder,
)
from coach_v1.test_coach_v1_task05_coach_encoder import _memory, _snapshot
from coach_v1.perception.team_perception import SpikeSharedInfo
from coach_v1.train_left_site_attack import build_trainer
from coach_v1.training.coach_environment import CoachTrainingEnvironment


class LeftSiteAttackTests(unittest.TestCase):
    def test_orb_and_ultimate_inputs_are_live_and_v2_stays_unchanged(self):
        orb = next((r, c) for r, line in enumerate(NEW_MAZE_STR.strip().splitlines())
                   for c, cell in enumerate(line) if cell == "5")
        base = _snapshot()
        allies = list(base.allies)
        allies[0] = dataclasses.replace(allies[0], ultimate_points=3, ultimate_cost=5)
        snapshot = dataclasses.replace(base, allies=tuple(allies), available_orbs=(orb,))
        belief = _memory().update(snapshot)
        old = CoachObservationEncoder().encode(snapshot, belief, situation="carry")
        new = CoachObservationEncoder(version=ORB_COACH_OBSERVATION_VERSION).encode(
            snapshot, belief, situation="carry")
        self.assertEqual((27, 26, 44), old.grid.shape)
        self.assertEqual((86,), old.vector.shape)
        self.assertEqual((28, 26, 44), new.grid.shape)
        self.assertEqual((96,), new.vector.shape)
        self.assertEqual(1, new.grid[ORB_COACH_GRID_CHANNELS.index("available_orb")][orb])
        self.assertAlmostEqual(0.3, new.vector[ORB_COACH_VECTOR_FIELDS.index(
            "slot_0_ultimate_points")])
        self.assertAlmostEqual(0.5, new.vector[ORB_COACH_VECTOR_FIELDS.index(
            "slot_0_ultimate_cost")])
        np.testing.assert_array_equal(old.grid, new.grid[:len(COACH_GRID_CHANNELS)])
        np.testing.assert_array_equal(
            old.vector,
            new.vector[[ORB_COACH_VECTOR_FIELDS.index(name) for name in COACH_VECTOR_FIELDS]],
        )
        self.assertEqual((5, 5), legal_action_mask(new).movement.shape)

    def test_left_curriculum_targets_only_left_site(self):
        environment = CoachTrainingEnvironment(
            Side.ATTACKER, seed=2100, stage="left_plant", max_ticks=10,
            observation_version=ORB_COACH_OBSERVATION_VERSION)
        for episode in range(3):
            state = environment.reset(episode=episode)
            self.assertLess(environment.game.target_plant_pos[1], 22)
            self.assertEqual(ORB_COACH_OBSERVATION_VERSION, state.observation.version)

    def test_candidate_loads_without_changing_official_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = build_trainer(directory=Path(directory), seed=2100)
            trainer.save()
            specialist = load_attacker_coach(Path(directory) / "latest.pt")
            incumbent = load_attacker_coach()
            policy = LeftSiteAttackPolicy(specialist, incumbent)
            self.assertEqual(ORB_COACH_OBSERVATION_VERSION, policy.encoder.version)
            self.assertEqual("coach-observation-v2", incumbent.encoder.version)
            before = _snapshot(tick=1)
            memory = _memory()
            policy.act(policy.encoder.encode(before, memory.update(before),
                                             situation="carry"))
            self.assertIsNone(incumbent.hidden)
            planted = dataclasses.replace(
                _snapshot(tick=2), spike=SpikeSharedInfo(None, None, True, (7, 3)))
            policy.act(policy.encoder.encode(planted, memory.update(planted),
                                             situation="guard"))
            self.assertIsNotNone(incumbent.hidden)


if __name__ == "__main__":
    unittest.main()
