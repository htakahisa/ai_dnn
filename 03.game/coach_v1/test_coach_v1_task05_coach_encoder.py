import dataclasses
from types import SimpleNamespace
import unittest

import numpy as np

from map_data import NEW_MAZE_STR

from coach_v1.common.checkpoint import build_checkpoint_metadata
from coach_v1.common.constants import FIXED_ROSTER
from coach_v1.common.types import Facing, ModelTarget, Side
from coach_v1.common.watch_points import load_watch_points
from coach_v1.common.constants import WATCH_POINTS_CONFIG_PATH
from coach_v1.observation import (
    AGE_CAP_TICKS,
    COACH_GRID_CHANNELS,
    COACH_VECTOR_FIELDS,
    CoachObservationEncoder,
    CoachObservationInputError,
)
from coach_v1.perception import (
    AllyPerception,
    BeliefMemory,
    EnemyPublicState,
    EnemySighting,
    PerceptionTick,
    SightingSource,
    SpikeSharedInfo,
    TeamPerceptionSnapshot,
)
from coach_v1.perception.team_perception import TeamPerceptionBuilder
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame


MAP = tuple(NEW_MAZE_STR.strip().splitlines())
CONFIG = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)


def _channel(observation, name):
    return observation.grid[COACH_GRID_CHANNELS.index(name)]


def _field(observation, name):
    return observation.vector[COACH_VECTOR_FIELDS.index(name)]


def _snapshot(tick=1, *, side=Side.ATTACKER, visible=(), sightings=(),
              smoke=(), spike=None, dead_slots=()):
    visible_set = set(visible)
    visible_grid = tuple(
        tuple((row, column) in visible_set for column in range(44))
        for row in range(26)
    )
    count_grid = tuple(tuple(int(value) for value in row) for row in visible_grid)
    allies = tuple(
        AllyPerception(
            slot=slot,
            character_id=roster.checkpoint_id,
            position=(23, 18 + slot),
            facing=Facing.E,
            is_alive=slot not in dead_slots,
            hp=0 if slot in dead_slots else 100,
            normal_ability_charges=0 if slot in dead_slots else 1,
            has_spike=False,
        )
        for slot, roster in enumerate(FIXED_ROSTER)
    )
    return TeamPerceptionSnapshot(
        side=side, tick=PerceptionTick(1, "live", tick), allies=allies,
        enemies=(EnemyPublicState("enemy", True),), sightings=tuple(sightings),
        currently_visible=visible_grid, visible_viewer_count=count_grid,
        smoke_cells=tuple(smoke),
        spike=spike or SpikeSharedInfo(None, None, False, None),
    )


def _memory(side=Side.ATTACKER):
    return BeliefMemory(CONFIG.for_side(side))


class CoachV1Task05EncoderTest(unittest.TestCase):
    def setUp(self):
        self.encoder = CoachObservationEncoder()

    def test_fixed_shape_dtype_metadata_and_static_map(self):
        snapshot = _snapshot()
        observation = self.encoder.encode(snapshot, _memory().update(snapshot), situation="carry")
        self.assertEqual((27, 26, 44), observation.grid.shape)
        self.assertEqual((84,), observation.vector.shape)
        self.assertEqual(np.dtype("float32"), observation.grid.dtype)
        self.assertEqual(np.dtype("float32"), observation.vector.dtype)
        self.assertFalse(observation.grid.flags.writeable)
        self.assertFalse(observation.vector.flags.writeable)
        self.assertEqual("coach-observation-v1", observation.version)
        self.assertEqual(CONFIG.map_hash, observation.map_hash)
        self.assertEqual(CONFIG.config_hash, observation.watch_points_hash)
        self.assertEqual(1.0, _channel(observation, "wall")[0, 0])
        self.assertEqual(0.0, _channel(observation, "walkable")[0, 0])
        self.assertEqual(1.0, _channel(observation, "attacker_spawn")[23, 18])
        self.assertEqual(1.0, _channel(observation, "plantable")[4, 41])
        self.assertTrue(np.isfinite(observation.grid).all())
        self.assertTrue(np.isfinite(observation.vector).all())

    def test_visibility_clear_and_watch_ages_are_separate_from_presence(self):
        point = CONFIG.for_side(Side.ATTACKER)[0]
        row, column = point.position
        memory = _memory()
        first = _snapshot(1, visible={point.position})
        first_obs = self.encoder.encode(first, memory.update(first), situation="carry")
        old = _snapshot(1 + AGE_CAP_TICKS + 10)
        old_obs = self.encoder.encode(old, memory.update(old), situation="carry")
        self.assertEqual(1.0, _channel(first_obs, "watch_importance")[row, column])
        self.assertEqual(1.0, _channel(first_obs, "currently_visible")[row, column])
        self.assertEqual(0.0, _channel(first_obs, "clear_age")[row, column])
        self.assertEqual(1.0, _channel(old_obs, "clear_known")[row, column])
        self.assertEqual(1.0, _channel(old_obs, "clear_age")[row, column])
        self.assertEqual(1.0, _channel(old_obs, "watch_confirmed")[row, column])
        self.assertEqual(1.0, _channel(old_obs, "watch_confirmation_age")[row, column])
        self.assertEqual(0.0, _channel(old_obs, "clear_known")[2, 18])
        self.assertEqual(0.0, _channel(old_obs, "clear_age")[2, 18])

    def test_last_seen_stays_at_reported_position_and_current_sighting_is_separate(self):
        memory = _memory()
        report = EnemySighting("enemy", (2, 17), 0, SightingSource.RECON)
        first = _snapshot(1, sightings=(report,))
        first_obs = self.encoder.encode(first, memory.update(first), situation="carry")
        hidden = _snapshot(5)
        hidden_obs = self.encoder.encode(hidden, memory.update(hidden), situation="carry")
        self.assertAlmostEqual(0.2, float(_channel(first_obs, "current_enemy_sighting")[2, 17]))
        self.assertEqual(0.0, _channel(hidden_obs, "current_enemy_sighting")[2, 17])
        self.assertAlmostEqual(0.2, float(_channel(hidden_obs, "last_seen_enemy_count")[2, 17]))
        self.assertEqual(1.0, _channel(hidden_obs, "last_seen_known")[2, 17])
        self.assertAlmostEqual(4 / AGE_CAP_TICKS, _channel(hidden_obs, "last_seen_age")[2, 17])

    def test_dead_slot_is_padded_without_shifting_later_slots(self):
        snapshot = _snapshot(dead_slots={1})
        observation = self.encoder.encode(snapshot, _memory().update(snapshot), situation="retrieve")
        self.assertEqual(0.0, _channel(observation, "ally_slot_1").sum())
        self.assertEqual(1.0, _channel(observation, "ally_slot_2")[23, 20])
        self.assertEqual(0.0, _field(observation, "slot_1_row"))
        self.assertEqual(0.0, _field(observation, "slot_1_facing_E"))
        self.assertEqual(1.0, _field(observation, "slot_2_alive"))
        self.assertAlmostEqual(0.8, float(_field(observation, "allies_alive")))
        self.assertEqual(1.0, _field(observation, "situation_retrieve"))

    def test_smoke_spike_phase_and_side(self):
        snapshot = _snapshot(
            side=Side.DEFENDER, smoke=((2, 17),),
            spike=SpikeSharedInfo(None, None, True, (4, 41)),
        )
        observation = self.encoder.encode(snapshot, _memory(Side.DEFENDER).update(snapshot), situation="retake")
        self.assertEqual(1.0, _channel(observation, "smoke")[2, 17])
        self.assertEqual(1.0, _channel(observation, "spike_planted")[4, 41])
        self.assertEqual(1.0, _field(observation, "spike_planted"))
        self.assertEqual(1.0, _field(observation, "side_defender"))
        self.assertEqual(1.0, _field(observation, "situation_retake"))

    def test_setup_and_dropped_spike_use_explicit_public_fields(self):
        snapshot = dataclasses.replace(
            _snapshot(side=Side.DEFENDER),
            tick=PerceptionTick(1, "defender_setup", 8),
            spike=SpikeSharedInfo(None, (4, 41), False, None),
        )
        observation = self.encoder.encode(
            snapshot, _memory(Side.DEFENDER).update(snapshot), situation="search"
        )
        self.assertEqual(1.0, _field(observation, "defender_setup"))
        self.assertEqual(1.0, _field(observation, "situation_search"))
        self.assertEqual(1.0, _field(observation, "spike_dropped"))
        self.assertEqual(1.0, _channel(observation, "spike_dropped")[4, 41])

    def test_rejects_mismatched_tick_side_phase_roster_shape_and_bad_age(self):
        snapshot = _snapshot()
        belief = _memory().update(snapshot)
        with self.assertRaises(CoachObservationInputError):
            self.encoder.encode(snapshot, belief, situation="search")
        with self.assertRaises(CoachObservationInputError):
            self.encoder.encode(dataclasses.replace(snapshot, tick=PerceptionTick(1, "live", 2)), belief, situation="carry")
        with self.assertRaises(CoachObservationInputError):
            self.encoder.encode(dataclasses.replace(snapshot, allies=tuple(reversed(snapshot.allies))), belief, situation="carry")
        bad_allies = (dataclasses.replace(snapshot.allies[0], character_id="wrong"),) + snapshot.allies[1:]
        with self.assertRaises(CoachObservationInputError):
            self.encoder.encode(dataclasses.replace(snapshot, allies=bad_allies), belief, situation="carry")
        with self.assertRaises(CoachObservationInputError):
            self.encoder.encode(snapshot, dataclasses.replace(belief, clear_age=belief.clear_age[:-1]), situation="carry")
        bad_age = list(map(list, belief.clear_age))
        bad_age[2][17] = -1
        with self.assertRaises(CoachObservationInputError):
            self.encoder.encode(snapshot, dataclasses.replace(belief, clear_age=tuple(map(tuple, bad_age))), situation="carry")

    def test_checkpoint_map_point_version_and_roster_match(self):
        metadata = build_checkpoint_metadata(
            target=ModelTarget.coach(Side.ATTACKER),
            map_hash=self.encoder.map_hash,
            watch_points_hash=self.encoder.watch_points_hash,
            model_config={}, training_seed=1, training_step=0,
        )
        self.encoder.validate_checkpoint(metadata, side=Side.ATTACKER)
        for change in (
            {"map_hash": "0" * 64},
            {"watch_points_hash": "0" * 64},
            {"observation_version": "other"},
            {"roster": tuple(reversed(metadata.roster))},
            {"target_id": "defender"},
        ):
            with self.subTest(change=change), self.assertRaises(CoachObservationInputError):
                self.encoder.validate_checkpoint(SimpleNamespace(**(vars(metadata) | change)), side=Side.ATTACKER)

    def test_hidden_enemy_truth_changes_do_not_change_end_to_end_actor_observation(self):
        class RejectReporter:
            def _blur_pos(self, *args, **kwargs):
                raise AssertionError("hidden enemy must not be reported")

        allies = [
            FakeCharacter(roster.character_name, "A", (23, 18 + slot), alive=slot == 0)
            for slot, roster in enumerate(FIXED_ROSTER)
        ]
        grid = np.asarray([[int(cell) for cell in row] for row in MAP], dtype=np.int8)
        builder = TeamPerceptionBuilder(iq_engine=RejectReporter())
        observations = []
        for hidden_position in ((2, 17), (2, 40)):
            enemy = FakeCharacter("hidden_enemy", "D", hidden_position)
            snapshot = builder.build(game=FakeGame(grid, allies + [enemy]), side=Side.ATTACKER)
            self.assertEqual((), snapshot.sightings)
            observations.append(self.encoder.encode(snapshot, _memory().update(snapshot), situation="carry"))
        np.testing.assert_array_equal(observations[0].grid, observations[1].grid)
        np.testing.assert_array_equal(observations[0].vector, observations[1].vector)


if __name__ == "__main__":
    unittest.main()
