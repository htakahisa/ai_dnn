"""Rally release, public utility candidates and snapshot isolation regressions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
import unittest
import numpy as np
from frc_v1.perception import FrcPerceptionBuilder
from frc_v1.actions import validate_action, build_masks
from touyama_v3.test.tv3_fixtures import defender_world as world
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_retake_coordination import RetakeAssembly
from touyama_v3.tv3_defender_policy import PolicyEncoder
from unittest.mock import patch


class RetakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = Scenario()

    def snapshot(self, side="L"):
        game = world(self.scenario)
        game.is_planted = True
        game.planted_pos = self.scenario.sites[side][0]
        return FrcPerceptionBuilder("D").build(game)





    def test_retake_combat_stops_shooting_and_does_not_defuse_under_uncovered_fire(self):
        from frc_v1.perception import Sighting
        from touyama_v3.tv3_retake_combat import RetakeEncoder, RETAKE_OBS_DIM
        from touyama_v3.tv3_defender_policy import DEFUSE_ACTION
        snapshot = self.snapshot()
        ally = replace(snapshot.allies[0], position=(7, 3), charges=0, facing='S')
        allies = (ally, *[replace(a, alive=False) for a in snapshot.allies[1:]])
        snapshot = replace(snapshot, allies=allies, sightings=(Sighting(0, (9, 3), 'normal'),), visible_cells=(), detonate_timer=40)
        encoder = RetakeEncoder(self.scenario)
        inputs = encoder.encode(snapshot, ally, snapshot.spike_planted, [.5, .5], {}, build_masks(snapshot))
        self.assertEqual(len(inputs.observation), RETAKE_OBS_DIM)
        self.assertTrue(inputs.mask[DEFUSE_ACTION])  # The policy remains free to attempt a defuse.
        self.assertNotEqual(inputs.teacher, DEFUSE_ACTION)
        self.assertEqual(inputs.actions[inputs.teacher].kind, 'STAY')
        self.assertEqual(inputs.actions[inputs.teacher].facing, 'S')
        urgent = replace(snapshot, detonate_timer=7)
        inputs = encoder.encode(urgent, ally, urgent.spike_planted, [.5, .5], {}, build_masks(urgent))
        self.assertEqual(inputs.teacher, DEFUSE_ACTION)

    def test_retake_victory_is_not_paid_to_already_dead_participant(self):
        from touyama_v3.tv3_retake_combat import apply_retake_outcome
        transitions = [[None, 0, -2., None, None, 1.], [None, 0, 0., None, None, 0.]]
        apply_retake_outcome(transitions, {'dead': 0, 'living': 1}, {'dead': False, 'living': True}, True)
        self.assertEqual(transitions[0][2], -2.)
        self.assertGreater(transitions[1][2], 0.)
        self.assertEqual(transitions[1][5], 1.)

    def test_started_defuse_is_not_cancelled_when_enemy_becomes_visible(self):
        from frc_v1.perception import Sighting
        from touyama_v3.tv3_retake_combat import RetakeEncoder
        from touyama_v3.tv3_defender_policy import DEFUSE_ACTION
        snapshot = self.snapshot()
        ally = replace(snapshot.allies[0], position=(8, 4), charges=0, defuse_progress=2)
        snapshot = replace(snapshot, allies=(ally, *[replace(a, alive=False) for a in snapshot.allies[1:]]),
            sightings=(Sighting(0, (9, 4), 'normal'),), visible_cells=(), detonate_timer=30)
        inputs = RetakeEncoder(self.scenario).encode(snapshot, ally, snapshot.spike_planted,
            [.5, .5], {}, build_masks(snapshot))
        self.assertEqual(inputs.teacher, DEFUSE_ACTION)
        self.assertTrue(inputs.mask[:40].any())  # Cancellation remains a learned decision.

    def test_repeated_start_cancel_cycles_do_not_farm_progress_reward(self):
        from types import SimpleNamespace as NS
        from frc_v1.actions import FrcAction
        from touyama_v3.tv3_retake_combat import deadline_action_reward
        inputs = NS(retake_context=dict(defuser=True, distance=0., slack=20))
        field = np.zeros((3, 3))
        start = deadline_action_reward(inputs, NS(defuse_progress=0),
            NS(defuse_timer=1, is_alive=True, pos=(1, 1)), FrcAction('DEFUSE'), field)
        stop = deadline_action_reward(inputs, NS(defuse_progress=1),
            NS(defuse_timer=0, is_alive=True, pos=(1, 1)), FrcAction('STAY'), field)
        self.assertLess(start+stop, 0.)

    def test_defuse_distance_includes_diagonals_and_role_does_not_switch_by_hp(self):
        from touyama_v3.tv3_retake_combat import RetakeRoles
        snapshot = self.snapshot()
        ally = replace(snapshot.allies[0], position=(8, 4), hp=100.)
        snapshot = replace(snapshot, allies=(ally, *[replace(a, alive=False) for a in snapshot.allies[1:]]))
        roles = RetakeRoles(self.scenario)
        selected, field = roles.select(snapshot)
        self.assertEqual(field[ally.position], 0)
        self.assertEqual(selected.slot, ally.slot)
        challenger = replace(snapshot.allies[1], alive=True, position=(7, 3), hp=100.)
        updated = replace(snapshot, allies=(replace(ally, hp=70.), challenger, *snapshot.allies[2:]))
        selected, _ = roles.select(updated)
        self.assertEqual(selected.slot, ally.slot)
        goals = roles.goals(updated, {ally.slot: (4, 8), challenger.slot: (7, 4)}, True)
        self.assertEqual(goals[ally.slot], ally.position)

    def test_progress_audit_records_choice_based_cancellation_and_available_idle(self):
        from types import SimpleNamespace as NS
        from frc_v1.actions import FrcAction
        from touyama_v3.tv3_retake_progress_audit import RetakeProgressAudit
        snapshot = self.snapshot()
        ally = replace(snapshot.allies[0], position=(8, 4), defuse_progress=2)
        snapshot = replace(snapshot, allies=(ally, *[replace(a, alive=False) for a in snapshot.allies[1:]]))
        data = NS(actions=(FrcAction('STAY'),), teacher=0, mask=np.ones(58, bool), combat_contact=False,
                  goal=ally.position, retake_context=dict(defuser=True))
        audit = RetakeProgressAudit(self.scenario)
        audit.before(snapshot, {ally.name: data}, {ally.name: ('retake', 0)}, True)
        char = NS(name=ally.name, team='D', pos=ally.position, defuse_timer=0, is_alive=True)
        audit.after(NS(chars=[char], is_defused=False))
        result = audit.report()
        self.assertEqual(result['counts']['interrupted_different_action'], 1)
        self.assertEqual(result['counts']['available_without_defuse_ticks'], 1)
        self.assertEqual(result['interruptions'][0]['lost_progress'], 2)
        data.actions = (FrcAction('DEFUSE'),)
        audit.before(snapshot, {ally.name: data}, {ally.name: ('retake', 0)}, True)
        char.is_alive, char.defuse_timer = False, 2  # Corpses can retain the engine timer.
        audit.after(NS(chars=[char], is_defused=False))
        self.assertEqual(audit.report()['counts']['interrupted_death'], 1)




    def test_retake_wait_is_bounded_when_two_members_are_ready(self):
        assembly = RetakeAssembly(self.scenario)
        snapshot = self.snapshot()
        goals = assembly.goals(snapshot)
        two = replace(snapshot, tick=snapshot.tick+7, allies=tuple(
            replace(a, position=goals[a.slot]) if a.slot < 2 else a for a in snapshot.allies))
        assembly.goals(two)
        self.assertTrue(assembly.launched)

    def test_retake_hidden_enemy_coordinates_do_not_change_public_features(self):
        from touyama_v3.tv3_retake_combat import RetakeEncoder
        game = world(self.scenario)
        game.is_planted, game.planted_pos = True, self.scenario.sites['L'][0]
        sensor = FrcPerceptionBuilder('D')
        first = sensor.build(game)
        a = first.allies[0]
        before = RetakeEncoder(self.scenario).encode(first, a, first.spike_planted, [.5, .5], {}, build_masks(first))
        for char in game.chars:
            if char.team == 'A':
                char.pos = [14, 36]
        second = sensor.build(game)
        after = RetakeEncoder(self.scenario).encode(second, second.allies[0], second.spike_planted,
            [.5, .5], {}, build_masks(second))
        np.testing.assert_array_equal(before.observation, after.observation)
        np.testing.assert_array_equal(before.mask, after.mask)
        self.assertEqual(before.teacher, after.teacher)

    def test_retake_checkpoint_requires_new_schema_while_search_stays_compatible(self):
        import tempfile
        import torch
        from pathlib import Path
        from touyama_v3.tv3_defender_controller import policy_metadata, load_policy
        from touyama_v3.tv3_defender_policy import DefenderDQN, OBS_DIM
        from touyama_v3.tv3_retake_combat import RETAKE_OBS_DIM
        from touyama_v3.tv3_retake_coordination import RETAKE_VERSION, retake_layout
        self.assertEqual(policy_metadata(self.scenario)['obs_dim'], OBS_DIM)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'retake.pt'
            saved = dict(phase='retake', schema=policy_metadata(self.scenario, 'retake'),
                retake_version=RETAKE_VERSION, retake_layout=retake_layout(self.scenario),
                model=DefenderDQN(RETAKE_OBS_DIM).state_dict())
            torch.save(saved, path)
            model, _ = load_policy(path, 'retake', self.scenario)
            self.assertEqual(model.obs_dim, RETAKE_OBS_DIM)
            torch.save({**saved, 'schema': policy_metadata(self.scenario)}, path)
            with self.assertRaises(ValueError):
                load_policy(path, 'retake', self.scenario)

    def test_rally_assignments_persist_and_all_ready_release_entry(self):
        snapshot = self.snapshot()
        # Initial search now includes a site position; exercise pre-entry rally from outside.
        snapshot = replace(snapshot, allies=tuple(replace(a, position=p) for a, p in zip(snapshot.allies,
            ((11, 1), (11, 2), (11, 3), (11, 4), (11, 5)))))
        assembly = RetakeAssembly(self.scenario)
        first = assembly.goals(snapshot)
        self.assertFalse(assembly.launched)
        self.assertEqual(len(set(first.values())), 5)
        second = assembly.goals(replace(snapshot, tick=snapshot.tick + 1))
        self.assertEqual(first, second)
        arrived = replace(snapshot, allies=tuple(replace(a, position=first[a.slot]) for a in snapshot.allies))
        goals = assembly.goals(arrived)
        self.assertTrue(assembly.launched)
        for ally in arrived.allies:
            self.assertIn(goals[ally.slot], [snapshot.spike_planted, *sum((list(v) for v in self.scenario.retake_entries['L'].values()), [])])

    def test_expiring_timer_and_single_survivor_release_without_waiting(self):
        snapshot = self.snapshot()
        assembly = RetakeAssembly(self.scenario)
        assembly.goals(replace(snapshot, detonate_timer=8))
        self.assertTrue(assembly.launched)
        alone = replace(snapshot, allies=tuple(replace(a, alive=a.slot == 0) for a in snapshot.allies))
        assembly = RetakeAssembly(self.scenario)
        self.assertEqual(set(assembly.goals(alone)), {0})
        self.assertTrue(assembly.launched)

    def test_dead_member_is_removed_from_rally_assignments(self):
        snapshot = self.snapshot()
        assembly = RetakeAssembly(self.scenario)
        assembly.goals(snapshot)
        dead = replace(snapshot, allies=tuple(replace(a, alive=False) if a.slot == 0 else a for a in snapshot.allies))
        self.assertNotIn(0, assembly.goals(dead))

    def test_every_active_utility_has_legal_candidates_without_lineups(self):
        snapshot = self.snapshot()
        for ability in ("FLASH", "RECON", "SMOKE", "RAMP", "ASH", "DANCE"):
            with self.subTest(ability=ability):
                allies = tuple(replace(a, ability_name=ability, charges=2, hp=40., max_hp=100.) if a.slot == 0
                               else replace(a, hp=40., max_hp=100.) for a in snapshot.allies)
                state = replace(snapshot, allies=allies)
                inputs = PolicyEncoder(self.scenario).encode(state, allies[0], snapshot.spike_planted, [.5, .5], {})
                candidates = np.flatnonzero(inputs.mask[40:48]) + 40
                self.assertGreater(len(candidates), 0)
                masks = build_masks(state)
                for index in candidates:
                    validate_action(state, masks, 0, inputs.actions[index])

    def test_non_target_ultimates_are_available_and_targets_are_legal(self):
        snapshot = self.snapshot()
        for ultimate in ("TUNNEL", "ESCAPE", "MONITOR", "RAID", "NEON", "BALEMOON"):
            with self.subTest(ultimate=ultimate):
                ally = replace(snapshot.allies[0], ultimate_name=ultimate, points=10, cost=1)
                state = replace(snapshot, allies=(ally, *snapshot.allies[1:]))
                inputs = PolicyEncoder(self.scenario).encode(state, ally, snapshot.spike_planted, [.5, .5], {})
                candidates = np.flatnonzero(inputs.mask[48:56]) + 48
                self.assertGreater(len(candidates), 0)
                for index in candidates:
                    validate_action(state, build_masks(state), 0, inputs.actions[index])


if __name__ == "__main__":
    unittest.main()
