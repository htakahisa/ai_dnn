"""Orb outing regression checks; never start training or load checkpoints."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ov1_defender_search_orb as orb
from ov1_ultimate_training import collect_orb_tick
import ov1_train_defender_search as training
import ov1_learning_defender_search as runtime


def unit(name, pos, points=0):
    return SimpleNamespace(name=name, pos=list(pos), team="D", is_alive=True,
                           ultimate_cost=6, ultimate_points=points,
                           orb_collect_timer=0, facing="S")


class OrbOutingTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((11, 11), dtype=np.int32)
        self.allies = [unit(str(i), (1, i + 2)) for i in range(5)]
        self.homes = {a.name: tuple(a.pos) for a in self.allies}
        self.orbs = {(5, 5)}
        self.outing = orb.IdleOrbOuting()
        self.patcher = patch.object(orb, "valid_orb_cells", return_value={(5, 5)})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def update(self, tick, **kwargs):
        return self.outing.update(self.grid, self.allies, self.homes, self.orbs,
                                  tick, 100 - tick, **kwargs)

    def start(self):
        self.assertFalse(self.update(1))
        self.assertTrue(self.update(1 + orb.IDLE_ORB_WAIT_TICKS))

    def test_single_collector_guards_and_return_home(self):
        self.start()
        self.assertEqual(len(set(self.outing.goals.values())), 5)
        collected_by = set()
        for tick in range(10, 90):
            active = self.update(tick)
            if active:
                for ally in self.allies:
                    occupied = {tuple(a.pos) for a in self.allies if a is not ally}
                    plan = self.outing.plan(ally, self.grid, occupied)
                    if plan.collect:
                        collected_by.add(ally.name)
                        collect_orb_tick(ally, self.orbs)
                    else:
                        ally.orb_collect_timer = 0
                        ally.pos[0] += plan.move[0]
                        ally.pos[1] += plan.move[1]
            if self.outing.phase == "done":
                break
        self.assertFalse(self.orbs)
        self.assertEqual(collected_by, {self.outing.collector})
        self.assertEqual(self.outing.phase, "done")
        self.assertEqual({a.name: tuple(a.pos) for a in self.allies}, self.homes)

    def test_contact_cancels_collection_even_after_arrival(self):
        self.start()
        collector = next(a for a in self.allies if a.name == self.outing.collector)
        collector.pos = [5, 5]
        collector.orb_collect_timer = 4
        self.assertFalse(self.update(10, threat=True))
        self.assertEqual(self.outing.phase, "returning")
        self.assertTrue(self.update(11))
        self.assertFalse(self.outing.plan(collector, self.grid).collect)
        self.assertEqual(self.orbs, {(5, 5)})

    def test_no_outing_during_setup_or_known_contact(self):
        self.assertFalse(self.update(1, setup=True))
        for tick in range(2, 30):
            self.assertFalse(self.update(tick, threat=True))
        self.assertEqual(self.outing.phase, "waiting")

    def test_full_ultimates_or_taken_orb_do_not_start(self):
        for ally in self.allies:
            ally.ultimate_points = ally.ultimate_cost
        self.update(1)
        self.assertFalse(self.update(10))
        self.assertEqual(self.outing.phase, "done")
        self.outing = orb.IdleOrbOuting()
        self.orbs.clear()
        self.update(1)
        self.assertFalse(self.update(10))

    def test_collector_death_aborts_and_threat_blocks_return(self):
        self.start()
        next(a for a in self.allies if a.name == self.outing.collector).is_alive = False
        self.assertFalse(self.update(10, threat=True))
        self.update(11)
        self.assertIn(self.outing.phase, {"returning", "done"})

    def test_flash_priority_and_fallback_when_full_or_dead(self):
        for status, expected in (("eligible", "0"), ("full", "1"), ("dead", "1")):
            with self.subTest(status=status):
                self.outing = orb.IdleOrbOuting()
                flash, non_flash = self.allies[:2]
                flash.ability_name = "FLASH"
                flash.ultimate_points = 0 if status != "full" else flash.ultimate_cost
                flash.is_alive = status != "dead"
                non_flash.ability_name = "SMOKE"
                non_flash.ultimate_points = non_flash.ultimate_cost - 1
                self.start()
                self.assertEqual(self.outing.collector, expected)

    def test_two_flash_units_use_ultimate_deficit_then_distance(self):
        for ally in self.allies[:2]:
            ally.ability_name = "FLASH"
        self.allies[0].ultimate_points = 1
        self.allies[1].ultimate_points = 0
        self.start()
        self.assertEqual(self.outing.collector, "0")
        self.outing = orb.IdleOrbOuting()
        self.allies[1].ultimate_points = 1
        self.start()
        self.assertEqual(self.outing.collector, "1")


class OrbIntegrationTests(unittest.TestCase):
    def make_env(self):
        env = training.SearchEnv()
        env.reset()
        env.in_setup_phase = False
        env.round_timer = training.MAX_TICKS - 10
        env.scheduled_smoke_fired = True
        env.idle_orb_outing.quiet_since = 0
        env.team_memory.reset()
        for attacker in env.attackers:
            attacker.is_alive = False
        # No dead carrier/ground-spike knowledge in the quiet-map fixture.
        env.attackers = []
        for d in env.defenders:
            d.pos = list(training.DEFENSE_ASSIGNMENT[d.name])
            d.ultimate_points = 0
        return env

    def test_real_map_collects_then_all_five_return(self):
        env = self.make_env()
        actions = {d.name: training.encode_action((0, 0), False, "N") for d in env.defenders}
        for _ in range(60):
            env.step(actions)
            if env.idle_orb_outing.phase == "done":
                break
        self.assertNotIn((16, 23), env.available_orbs)
        self.assertEqual(env.idle_orb_outing.phase, "done")
        self.assertTrue(all(tuple(d.pos) == training.DEFENSE_ASSIGNMENT[d.name]
                            for d in env.defenders))

    def test_known_spike_cancels_in_progress_collection(self):
        env = self.make_env()
        env.idle_orb_outing.update(training.GRID, env.defenders, training.DEFENSE_ASSIGNMENT,
                                  env.available_orbs, 11, env.round_timer)
        collector = next(d for d in env.defenders if d.name == env.idle_orb_outing.collector)
        collector.pos = list(env.idle_orb_outing.orb)
        collector.orb_collect_timer = 4
        env.team_memory.spike_pos = (16, 22)
        env.team_memory.spike_held = False
        actions = {d.name: training.encode_action((0, 0), False, "N", use_orb=True)
                   for d in env.defenders}
        _, _, _, _, actual = env.step(actions)
        self.assertIn((16, 23), env.available_orbs)
        self.assertEqual(collector.orb_collect_timer, 0)
        self.assertTrue(all(not training.decode_action(a)[3] for a in actual.values()))

    def test_training_executes_the_shared_plan_and_records_it(self):
        env = self.make_env()
        actions = {d.name: training.encode_action((0, 0), False, "N") for d in env.defenders}
        start_positions = {d.name: tuple(d.pos) for d in env.defenders}
        _, _, _, _, actual = env.step(actions)
        self.assertEqual(env.idle_orb_outing.phase, "outbound")
        self.assertEqual(env._orb_plan_names, set(start_positions))
        self.assertEqual(env.idle_orb_outing.orb, (16, 23))
        for d in env.defenders:
            move, ability, ultimate, collect, facing = training.decode_action(actual[d.name])
            self.assertEqual(tuple(d.pos), tuple(start_positions[d.name][i] + move[i] for i in range(2)))
            self.assertFalse(ability or ultimate or collect)

    def test_runtime_uses_same_plan_without_invoking_policy(self):
        env = self.make_env()
        ctrl = object.__new__(runtime.Ov1LearningDefenderSearchController)
        ctrl.idle_orb_outing = orb.IdleOrbOuting()
        ctrl.idle_orb_outing.quiet_since = 0
        ctrl.team_memory = env.team_memory
        ctrl._site_positions_cache = [(7, 3), (7, 40)]
        ctrl._assigned_positions = dict(training.DEFENSE_ASSIGNMENT)
        for method in ("_ensure_defense_assignment", "_maybe_advance_tick",
                       "_update_priority_dist_maps", "_update_right_site_rotation"):
            setattr(ctrl, method, Mock())
        ctrl._scheduled_smoke_target = Mock(return_value=None)
        ctrl._build_observation = Mock(return_value=(np.zeros(training.OBS_DIM), []))
        d = env.defenders[0]
        occupied = {tuple(a.pos) for a in env.defenders if a is not d}
        reference = orb.IdleOrbOuting()
        reference.quiet_since = 0
        reference.update(training.GRID, env.defenders, training.DEFENSE_ASSIGNMENT,
                         env.available_orbs, 11, env.round_timer)
        expected = reference.plan(d, training.GRID, occupied)
        result = ctrl.decide_move(d, {
            "grid": training.GRID, "chars": env.defenders, "battle_tick": 10,
            "round_timer": env.round_timer, "available_orbs": env.available_orbs,
        })
        self.assertEqual(result[0], [d.pos[0] + expected.move[0], d.pos[1] + expected.move[1]])
        self.assertEqual(result[1]["facing"], expected.facing)


if __name__ == "__main__":
    unittest.main()
