import dataclasses
import unittest

import numpy as np

from map_data import NEW_MAZE_STR
from coach_v1.common.checkpoint import build_checkpoint_metadata
from coach_v1.common.constants import FIXED_ROSTER, WATCH_POINTS_CONFIG_PATH
from coach_v1.common.types import Facing, ModelTarget, MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.common.watch_points import load_watch_points
from coach_v1.observation import CharacterInputError, CharacterObservationEncoder, CoachInstruction
from coach_v1.perception import BeliefMemory, TeamPerceptionBuilder
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.test_coach_v1_task05_coach_encoder import _memory, _snapshot
from coach_v1.training import CharacterAction, CharacterEnvironment, CurriculumStage, ScenarioGenerator


class CharacterEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.environment = CharacterEnvironment()
        self.instruction = CoachInstruction(MovementAction.MOVE_E, ObjectiveAction.NONE,
                                             TacticalIntent.CLEAR_AREA)

    def prepare(self, slot=0, snapshot=None, instruction=None, curriculum=None):
        snapshot = snapshot or _snapshot()
        memory = _memory(snapshot.side)
        belief = memory.update(snapshot)
        return self.environment.prepare(snapshot, belief,
                                        situation="carry" if snapshot.side is Side.ATTACKER else "search",
                                        slot=slot, instruction=instruction or self.instruction,
                                        curriculum=curriculum)

    def test_shape_external_move_and_facing_only(self):
        step = self.prepare()
        obs = step.observation
        self.assertEqual((29, 26, 44), obs.grid.shape)
        self.assertEqual((106,), obs.vector.shape)
        self.assertEqual("character-observation-v1", obs.version)
        self.assertFalse(obs.grid.flags.writeable)
        self.assertFalse(obs.mask.ability_target.flags.writeable)
        self.assertTrue(np.isfinite(obs.grid).all())
        self.assertTrue(np.isfinite(obs.vector).all())
        self.assertEqual(((23, 19), {"facing": "NW"}),
                         self.environment.resolve(step, CharacterAction(Facing.NW)))
        stay = self.prepare(instruction=dataclasses.replace(
            self.instruction, movement=MovementAction.STAY))
        self.assertEqual(((23, 18), {"facing": "S"}),
                         self.environment.resolve(stay, CharacterAction(Facing.S)))
        self.assertFalse(hasattr(CharacterAction(Facing.N), "movement"))

    def test_ability_stops_movement_and_rejects_invalid_target(self):
        step = self.prepare()
        self.assertEqual("SMOKE", step.ability_name)
        self.assertTrue(step.observation.mask.ability_target[23, 18])
        self.assertFalse(step.observation.mask.ability_target[0, 0])
        self.assertEqual(((23, 18), {"ability": "SMOKE", "target": (23, 18), "facing": "N"}),
                         self.environment.resolve(step, CharacterAction(Facing.N, True, (23, 18))))
        for target in ((0, 0), (-1, 0), (26, 0), (23, 18.5)):
            with self.subTest(target=target), self.assertRaises(CharacterInputError):
                self.environment.resolve(step, CharacterAction(Facing.N, True, target))

    def test_character_ability_differences_and_charges(self):
        self.assertFalse(self.prepare(slot=1).observation.mask.ability_use[1])  # HUNT has no active action
        for slot, ability in ((2, "RECON"), (3, "FLASH"), (4, "FLASH")):
            step = self.prepare(slot=slot)
            self.assertEqual(ability, step.ability_name)
            self.assertFalse(step.observation.mask.ability_target[23, 18 + slot])
            self.assertTrue(step.observation.mask.ability_use[1])
        snapshot = _snapshot()
        allies = list(snapshot.allies)
        allies[0] = dataclasses.replace(allies[0], normal_ability_charges=0)
        empty = self.prepare(snapshot=dataclasses.replace(snapshot, allies=tuple(allies)))
        self.assertFalse(empty.observation.mask.ability_use[1])
        with self.assertRaises(CharacterInputError):
            self.environment.resolve(empty, CharacterAction(Facing.N, True, (23, 18)))

    def test_projectile_wall_and_objective_priority(self):
        step = self.prepare(slot=2)
        self.assertFalse(step.observation.mask.ability_target[0, 0])
        map_rows = tuple(NEW_MAZE_STR.strip().splitlines())
        game = FakeGame([[int(cell) for cell in row] for row in map_rows], [])
        game.height, game.width = 26, 44
        for target in ((23, 20), (22, 20), (21, 20), (20, 20), (22, 17), (19, 19)):
            row, column = target
            legal = map_rows[row][column] != "1" and len(game._projectile_path(step.position, target)) > 1
            self.assertEqual(legal, bool(step.observation.mask.ability_target[target]))
        objective = self.prepare(instruction=dataclasses.replace(
            self.instruction, objective=ObjectiveAction.PLANT))
        self.assertFalse(objective.observation.mask.ability_use[1])
        self.assertEqual(((23, 18), "PLANT"),
                         self.environment.resolve(objective, CharacterAction(Facing.E)))
        with self.assertRaises(CharacterInputError):
            self.environment.resolve(objective, CharacterAction(Facing.E, True, (23, 18)))

    def test_dead_actor_and_curriculum(self):
        dead = self.prepare(slot=0, snapshot=_snapshot(dead_slots={0}))
        self.assertFalse(dead.observation.mask.facing.any())
        self.assertFalse(dead.observation.mask.ability_use.any())
        with self.assertRaises(CharacterInputError):
            self.environment.resolve(dead, CharacterAction(Facing.N))
        one = _snapshot(dead_slots={1, 2, 3, 4})
        self.prepare(snapshot=one, curriculum=CurriculumStage.ONE_V_ONE)
        with self.assertRaises(CharacterInputError):
            self.prepare(snapshot=one, curriculum=CurriculumStage.TWO_V_ONE)
        two = _snapshot(dead_slots={2, 3, 4})
        self.prepare(snapshot=two, curriculum=CurriculumStage.TWO_V_ONE)
        with self.assertRaises(CharacterInputError):
            self.prepare(slot=2, snapshot=two, curriculum=CurriculumStage.TWO_V_ONE)

    def test_checkpoint_metadata_is_bound_to_character_and_fixed_map(self):
        encoder = CharacterObservationEncoder()
        observation = self.prepare().observation
        metadata = build_checkpoint_metadata(
            target=ModelTarget.character("gorimaru"), map_hash=observation.map_hash,
            watch_points_hash=observation.watch_points_hash, model_config={},
            training_seed=1, training_step=0,
        )
        encoder.validate_checkpoint(metadata, slot=0)
        with self.assertRaises(CharacterInputError):
            encoder.validate_checkpoint(metadata, slot=1)
        with self.assertRaises(CharacterInputError):
            encoder.validate_checkpoint(dataclasses.replace(metadata, map_hash="0" * 64), slot=0)

    def test_hidden_enemy_positions_do_not_change_actor_observation_or_mask(self):
        map_rows = tuple(NEW_MAZE_STR.strip().splitlines())
        grid = np.asarray([[int(cell) for cell in row] for row in map_rows], dtype=np.int8)
        allies = [FakeCharacter(roster.character_name, "A", (23, 18 + slot),
                                alive=slot == 0) for slot, roster in enumerate(FIXED_ROSTER)]
        builder = TeamPerceptionBuilder()
        baseline = builder.build(game=FakeGame(grid, allies), side=Side.ATTACKER)
        generator = ScenarioGenerator(13)
        scenarios = [generator.generate(
            actor_side="attacker", situation="carry", elapsed_ticks=100, enemy_count=1,
            currently_visible=baseline.currently_visible,
            occupied_positions=(ally.position for ally in baseline.allies)) for _ in range(2)]
        self.assertNotEqual(scenarios[0].enemies[0].position, scenarios[1].enemies[0].position)
        config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        outputs = []
        for scenario in scenarios:
            enemy = FakeCharacter("hidden", "D", scenario.enemies[0].position)
            snapshot = builder.build(game=FakeGame(grid, allies + [enemy]), side=Side.ATTACKER)
            self.assertEqual((), snapshot.sightings)
            belief = BeliefMemory(config.for_side(Side.ATTACKER)).update(snapshot)
            outputs.append(CharacterObservationEncoder().encode(
                snapshot, belief, situation="carry", slot=0, instruction=self.instruction))
        np.testing.assert_array_equal(outputs[0].grid, outputs[1].grid)
        np.testing.assert_array_equal(outputs[0].vector, outputs[1].vector)
        np.testing.assert_array_equal(outputs[0].mask.ability_target, outputs[1].mask.ability_target)


if __name__ == "__main__":
    unittest.main()
