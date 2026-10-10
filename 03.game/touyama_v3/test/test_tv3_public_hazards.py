"""Public hazard observations and policy compatibility; no training runs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
import tempfile
import unittest
import numpy as np
import torch

from public_effects import DisplayEffect
from frc_v1.perception import FrcPerceptionBuilder
from frc_v1.actions import build_masks
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.test.tv3_fixtures import defender_world, attacker_world
from touyama_v3.tv3_learn_public_hazards import hazard_features, HAZARD_OBS_DIM
from touyama_v3.tv3_defender_policy import PolicyEncoder, OBS_DIM, ACTION_DIM
from touyama_v3.tv3_defender_controller import load_policy, policy_metadata


class PublicHazardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = Scenario()

    def setUp(self):
        self.snapshot = FrcPerceptionBuilder('D').build(defender_world(self.scenario))
        self.ally = self.snapshot.allies[0]
        self.goal = self.ally.position

    def effect(self, kind='NEON', phase='active', cells=None, level=0):
        return DisplayEffect(1, kind, phase, self.ally.position,
                             cells=tuple(cells or (self.ally.position,)), level=level)

    def encode(self, effects=()):
        return hazard_features(replace(self.snapshot, effects=effects), self.ally, self.goal)

    def test_warning_active_and_disappearance_are_distinguishable(self):
        warning = self.encode((self.effect(phase='warning'),))
        active = self.encode((self.effect(),))
        # 49 local cells followed by STAY,N,E,S,W in four-channel order.
        np.testing.assert_array_equal(warning[196:200], [1, 0, 0, 0])
        np.testing.assert_array_equal(active[196:200], [0, 1, 0, 0])
        self.assertEqual(len(active), HAZARD_OBS_DIM)
        self.assertFalse(self.encode().any())

    def test_destination_and_distant_goal_and_spike_have_hazard_features(self):
        r, c = self.ally.position
        east = (r, c + 1)
        distant = (r + 5, c)
        snapshot = replace(self.snapshot, effects=(self.effect(cells=(east, distant)),),
                           spike_dropped=distant)
        features = hazard_features(snapshot, self.ally, distant)
        np.testing.assert_array_equal(features[204:208], [0, 1, 0, 0])
        np.testing.assert_array_equal(features[216:224], [0, 1, 0, 0] * 2)

    def test_destruction_level_and_balemoon_warning(self):
        features = self.encode((self.effect('DESTRUCTION', level=5),
                                self.effect('BALEMOON', 'warning')))
        np.testing.assert_array_equal(features[196:200], [0, 0, .5, 1])

    def test_duplicates_order_and_out_of_bounds_do_not_change_features(self):
        effect = self.effect()
        invalid = self.effect(cells=((-1, -1), (999, 999)))
        np.testing.assert_array_equal(self.encode((effect,)), self.encode((invalid, effect, effect)))

    def test_flight_and_nonhazard_effects_do_not_invent_impact_cells(self):
        self.assertFalse(self.encode((self.effect('ASH', 'flight'), self.effect('FLASH'),
                                     self.effect('SMOKE'))).any())

    def test_own_contract_status_is_distinct_from_standing_in_an_area(self):
        ally = replace(self.ally, contract=5, max_hp_lost=20)
        features = hazard_features(self.snapshot, ally, self.goal)
        np.testing.assert_allclose(features[-3:], [1, .5, .2])
        self.assertFalse(features[:-3].any())

    def test_policy_observes_hazards_without_forbidding_moves_or_wait(self):
        encoder = PolicyEncoder(self.scenario)
        baseline = encoder.encode(self.snapshot, self.ally, self.goal, [.5, .5], {})
        snapshot = replace(self.snapshot, effects=(self.effect(),))
        exposed = encoder.encode(snapshot, self.ally, self.goal, [.5, .5], {})
        self.assertEqual(len(exposed.observation), OBS_DIM)
        self.assertFalse(np.array_equal(baseline.observation, exposed.observation))
        np.testing.assert_array_equal(baseline.mask, exposed.mask)
        np.testing.assert_array_equal(baseline.observation[:505], exposed.observation[:505])

    def test_old_search_checkpoint_is_rejected_before_loading_weights(self):
        schema = {**policy_metadata(self.scenario), 'version': 2, 'obs_dim': 505}
        schema.pop('public_hazards')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            torch.save(dict(phase='search', schema=schema, model={}), path)
            with self.assertRaisesRegex(ValueError, 'checkpoint/schema mismatch'):
                load_policy(path, 'search', self.scenario)

    def test_old_plant_and_guard_schemas_are_rejected(self):
        from touyama_v3.tv3_learn_attacker_plant import policy_schema, load_plant
        from touyama_v3.tv3_learn_attacker_guard import guard_schema, load_guard
        with tempfile.TemporaryDirectory() as directory:
            for name, factory, loader in (('plant', policy_schema, load_plant),
                                           ('guard', guard_schema, load_guard)):
                with self.subTest(policy=name):
                    schema = factory(self.scenario)
                    schema.pop('public_hazards')
                    schema['obs_dim'] -= HAZARD_OBS_DIM
                    schema['version'] -= 1
                    path = Path(directory) / (name + '.pt')
                    torch.save(dict(schema=schema, model={}), path)
                    with self.assertRaisesRegex(ValueError, 'schema mismatch'):
                        loader(path, self.scenario)

    def test_plant_retake_and_guard_receive_shared_hazard_features(self):
        from touyama_v3.tv3_learn_attacker_plant import PlantEncoder, OBS_DIM as PLANT_DIM
        from touyama_v3.tv3_learn_attacker_guard import GuardEncoder, OBS_DIM as GUARD_DIM
        from touyama_v3.tv3_retake_combat import RetakeEncoder, RETAKE_OBS_DIM
        from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel
        from touyama_v3.tv3_collect_attacker_analysis import AnalysisCollectorController
        game = attacker_world()
        encoder = AttackerEncoder(game)
        model = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(game.names))
        controller = AnalysisCollectorController(game, model, np.random.default_rng(0))
        controller.set_game(game)
        controller.prepare_team_tick()
        snapshot = controller.snapshot
        ally = snapshot.allies[0]
        route = controller.route
        belief = model.analyze(encoder, snapshot, controller.frames[0]['observation'], [route])
        plant_encoder = PlantEncoder(game)
        plain = plant_encoder.encode(snapshot, ally, route.cells[-1], route, belief, {}, build_masks(snapshot), 0, 'supported')
        snapshot = replace(snapshot, effects=(replace(self.effect(), position=ally.position, cells=(ally.position,)),))
        exposed = plant_encoder.encode(snapshot, ally, route.cells[-1], route, belief, {}, build_masks(snapshot), 0, 'supported')
        self.assertEqual(len(exposed.observation), PLANT_DIM)
        self.assertFalse(np.array_equal(plain.observation, exposed.observation))
        np.testing.assert_array_equal(plain.mask[:40], exposed.mask[:40])
        planted = replace(self.snapshot, is_planted=True, spike_planted=self.scenario.sites['L'][0],
                          effects=(self.effect(),))
        retake = RetakeEncoder(self.scenario).encode(planted, self.ally, planted.spike_planted, [.5, .5], {}, build_masks(planted))
        self.assertEqual(len(retake.observation), RETAKE_OBS_DIM)
        np.testing.assert_array_equal(retake.observation[505:OBS_DIM], hazard_features(planted, self.ally, planted.spike_planted))
        guard_snapshot = replace(snapshot, is_planted=True, spike_planted=game.sites['L'][0])
        guard = GuardEncoder(game).encode(guard_snapshot, ally, guard_snapshot.spike_planted, {},
                    build_masks(guard_snapshot), 0, 55, 5, 5, sum(a.charges for a in snapshot.allies))
        self.assertEqual(len(guard.observation), GUARD_DIM)
        self.assertEqual(guard.observation[197 + 505], 1.)


if __name__ == '__main__':
    unittest.main()
