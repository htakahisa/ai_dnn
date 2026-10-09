"""Synthetic entry support checks; no training matches or model files."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from abilities_los import AbilityLosMixin
from frc_v1.actions import KINDS, build_masks
from toruAI_v4.tv4_attacker_entry_utility import (predicted_entry_utility, flash_entry_step, entry_support_pending,
                                              SupportUseTracker, SUPPORT_REFRESH_TICKS)
from toruAI_v4.tv4_collect_attacker_analysis import AnalysisCollectorController
from toruAI_v4.tv4_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel, Route
from toruAI_v4.tv4_learn_attacker_plant import PlantEncoder, PlantDQN
from toruAI_v4.tv4_attacker_plant_controller import ToruV4AttackerPlantController
from toruAI_v4.tv4_scenario import Scenario
from toruAI_v4.test.test_tv4_attacker_analysis import world


class Room(AbilityLosMixin):
    def __init__(self):
        self.grid = np.zeros((11, 11), dtype=int)
        self.grid[0, :] = self.grid[-1, :] = 1
        self.grid[:, 0] = self.grid[:, -1] = 1
        self.grid[:, 8] = 1
        self.height, self.width = self.grid.shape
        self.names = ("site", "approach")
        self.region = np.where(np.indices(self.grid.shape)[1] >= 5, 0, 1)

    def clear(self, a, b):
        return all(self.grid[p] != 1 for p in self._line_cells(a, b))

    def neighbors(self, p):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = p[0] + dr, p[1] + dc
            if 0 <= q[0] < self.height and 0 <= q[1] < self.width and self.grid[q] != 1:
                yield q


class EntryUtilityTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Room()
        self.ally = SimpleNamespace(slot=0, ability_name="FLASH", position=(5, 2))
        self.snapshot = SimpleNamespace(visible_cells=(), sightings=(), smoke_cells=(), effects=())
        self.route = Route("L", tuple((5, c) for c in range(2, 8)), ("approach", "site"))
        kinds = np.ones((5, len(KINDS)), dtype=bool)
        targets = np.tile((self.scenario.grid != 1).ravel(), (5, 2, 1))
        self.masks = SimpleNamespace(kind=kinds, target=targets)
        self.belief = {"trained_rounds": 120, "placement": {
            i: {"site": .95, "approach": .05, "dead": 0.} for i in range(5)}}

    def plan(self):
        return predicted_entry_utility(self.scenario, self.snapshot, self.ally,
                                       self.belief, self.route, self.masks)

    def test_unseen_flash_uses_actual_wall_stopped_impact(self):
        cast, delay = self.plan()
        self.assertEqual(cast["ability"], "FLASH")
        path = self.scenario._projectile_path(self.ally.position, cast["target"])
        self.assertGreater(delay, 0)
        self.assertLessEqual(delay, 5)
        self.assertEqual(self.scenario.grid[path[-1]], 0)
        self.assertTrue(self.scenario.clear(path[-1], (5, 6)))
        self.assertTrue(self.masks.target[0, 0, cast["target"][0] * 11 + cast["target"][1]])

    def test_untrained_or_uncertain_belief_does_not_spend_utility(self):
        self.belief["trained_rounds"] = 0
        self.assertIsNone(self.plan())
        self.belief["trained_rounds"] = 120
        for row in self.belief["placement"].values():
            row.update(site=.5, approach=.5)
        self.assertIsNone(self.plan())

    def test_empty_visible_region_and_illegal_ability_are_excluded(self):
        self.snapshot.visible_cells = tuple(map(tuple, np.argwhere(self.scenario.grid != 1)))
        self.assertIsNone(self.plan())
        self.snapshot.visible_cells = ()
        self.masks.kind[0, KINDS.index("ABILITY")] = False
        self.assertIsNone(self.plan())

    def test_active_utility_prevents_duplicate_cast(self):
        self.snapshot.effects = (SimpleNamespace(kind="FLASH", phase="flight"),)
        self.assertIsNone(self.plan())

    def test_smoke_and_recon_also_support_unseen_entry(self):
        for kind in ("SMOKE", "RECON", "ASH"):
            self.ally.ability_name = kind
            cast, delay = self.plan()
            self.assertEqual(cast["ability"], kind)
            self.assertGreaterEqual(delay, 0)

    def test_real_scenario_supports_unmocked_projectile_planning(self):
        scenario = Scenario()
        corridor = next(tuple((r, c + i) for i in range(5))
                        for r, c in map(tuple, np.argwhere(scenario.grid != 1))
                        if c + 4 < scenario.grid.shape[1]
                        and all(scenario.grid[r, c + i] != 1 for i in range(5)))
        region = scenario.names[int(scenario.region[corridor[-1]])]
        route = Route("L", corridor, (region,))
        masks = SimpleNamespace(kind=np.ones((5, len(KINDS)), dtype=bool),
            target=np.tile((scenario.grid != 1).ravel(), (5, 2, 1)))
        belief = {"trained_rounds": 120, "placement": {
            i: {n: float(n == region) for n in scenario.names} for i in range(5)}}
        # Confirm the distant parts of this synthetic region empty. The planner
        # must not pretend a broad region belief locates the enemy at its nearest cell.
        snapshot = SimpleNamespace(**vars(self.snapshot))
        snapshot.visible_cells = tuple(tuple(map(int, p)) for p in np.argwhere(scenario.grid != 1)
                                      if max(abs(int(p[0])-corridor[-1][0]), abs(int(p[1])-corridor[-1][1])) > 4)
        for kind in ("FLASH", "RECON", "SMOKE"):
            ally = SimpleNamespace(slot=0, ability_name=kind, position=corridor[0])
            cast, delay = predicted_entry_utility(scenario, snapshot, ally, belief, route, masks)
            self.assertEqual(cast["ability"], kind)
            if kind in ("FLASH", "RECON"):
                self.assertGreater(delay, 0)
                path = scenario._projectile_path(ally.position, cast["target"])
                self.assertGreater(len(path), 1)
                self.assertTrue(all(scenario.grid[p] != 1 for p in path))

    def test_collector_recon_does_not_stop_teammates_or_repeat_cast(self):
        game = world()
        game.chars[0].ability_name = "RECON"
        game.chars[0].recon_charges = 2
        game.chars[0].flash_charges = 0
        scenario = Scenario()
        encoder = AttackerEncoder(scenario)
        model = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(scenario.names))
        model.trained_rounds = 120
        controller = AnalysisCollectorController(scenario, model, np.random.default_rng(0))
        controller.set_game(game)
        controller.prepare_team_tick()
        controller.mode = "supported"
        caster = next(a for a in controller.snapshot.allies if a.ability_name == "RECON")
        plan = ({"ability": caster.ability_name, "target": controller.route.cells[-1]}, 3)
        def select_cast(_scenario, snapshot, ally, *args):
            return plan if ally.slot == caster.slot else None
        with patch("toruAI_v4.tv4_collect_attacker_analysis.predicted_entry_utility", side_effect=select_cast):
            for tick in (1, 2, 3):
                game.battle_tick = tick
                controller.prepare_team_tick()
                self.assertTrue(any(tuple(controller.actions[a.name][0]) != a.position
                                    for a in controller.snapshot.allies if a.slot != caster.slot))
                if tick == 1:
                    self.assertEqual(tuple(controller.actions[caster.name][0]), caster.position)
            self.assertEqual(controller.entry_wait_until, 1)
            game.battle_tick = 4
            controller.prepare_team_tick()
            self.assertTrue(any(tuple(controller.actions[a.name][0]) != a.position
                                for a in controller.snapshot.allies))
        self.assertEqual(sum(e["type"] == "predicted_entry_utility" for e in controller.events), 1)

    def test_plant_candidate_and_teacher_use_predicted_cast_without_forcing_runtime(self):
        game = world()
        scenario = Scenario()
        encoder = AttackerEncoder(scenario)
        model = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(scenario.names))
        controller = AnalysisCollectorController(scenario, model, np.random.default_rng(0))
        controller.set_game(game)
        controller.prepare_team_tick()
        snapshot = controller.snapshot
        ally = snapshot.allies[0]
        legal = build_masks(snapshot)
        flat = int(np.flatnonzero(legal.target[0, 0])[0])
        target = divmod(flat, scenario.grid.shape[1])
        belief = model.analyze(encoder, snapshot, controller.frames[0]["observation"], [controller.route])
        with patch("toruAI_v4.tv4_learn_attacker_plant.predicted_entry_utility",
                   return_value=({"ability": ally.ability_name, "target": target}, 2)):
            inputs = PlantEncoder(scenario).encode(snapshot, ally, controller.route.cells[-1],
                        controller.route, belief, {}, legal, 0, "supported")
            self.assertEqual(inputs.actions[43].target, target)
            self.assertTrue(inputs.mask[43])
            self.assertEqual(inputs.teacher, 43)
            flight = SimpleNamespace(kind="FLASH", phase="flight", position=ally.position)
            inputs = PlantEncoder(scenario).encode(replace(snapshot, effects=(flight,)), ally,
                        controller.route.cells[-1], controller.route, belief, {}, legal, 0, "supported")
            self.assertNotEqual(inputs.actions[inputs.teacher].kind, "STAY")

    def test_plant_progress_mask_keeps_waiting_legal_during_flight(self):
        game = world()
        encoder = AttackerEncoder(game)
        model = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(game.names))
        controller = ToruV4AttackerPlantController(game, model, PlantDQN(), training=True)
        controller.teacher_probability = 1.
        controller.set_game(game)
        controller.prepare_team_tick()
        controller.route_mode = "supported"
        ally = controller.snapshot.allies[0]
        flight = SimpleNamespace(kind="FLASH", phase="flight", position=ally.position)
        snapshot = replace(controller.snapshot, tick=1, sightings=(), effects=(flight,))
        game.battle_tick = 1
        with patch.object(controller.sensor, "build", return_value=snapshot):
            controller.prepare_team_tick()
        inputs = controller.inputs[ally.name]
        self.assertTrue(inputs.mask[inputs.teacher])
        self.assertTrue(inputs.mask[:8].any())  # Waiting remains a learned choice.
        self.assertNotEqual(inputs.actions[inputs.teacher].kind, "STAY")

    def test_flash_boundary_waits_behind_cover_and_leaves_exposed_cells(self):
        self.scenario.grid[5, 4] = 1
        impact = (5, 6)
        self.assertEqual(flash_entry_step(self.scenario, (5, 3), (4, 3), impact), (5, 3))
        step = flash_entry_step(self.scenario, (4, 3), (4, 4), impact)
        self.assertNotEqual(step, (4, 3))
        self.assertFalse(self.scenario.clear(step, impact))
        self.assertEqual(flash_entry_step(self.scenario, (5, 2), (5, 3), impact), (5, 3))

    def test_recon_flight_does_not_trigger_entry_wait(self):
        ally = self.ally
        snapshot = SimpleNamespace(effects=(SimpleNamespace(kind="RECON", phase="flight", position=ally.position),))
        self.assertFalse(entry_support_pending(snapshot, ally))

    def test_seen_region_is_not_used_as_hidden_enemy_confidence(self):
        self.snapshot.sightings = tuple(SimpleNamespace(enemy_id=i, position=(5, 6)) for i in range(5))
        self.assertIsNone(self.plan())

    def test_repeat_support_after_progress_contact_or_effect_expiry(self):
        holder = SimpleNamespace(alive=True, has_spike=True, position=(5, 2))
        snapshot = SimpleNamespace(tick=1, allies=(holder,), spike_dropped=None, effects=(), sightings=())
        tracker = SupportUseTracker()
        for kind in ("FLASH", "SMOKE", "RECON"):
            target = (5, 6)
            tracker.record(snapshot, {"ability": kind, "target": target})
            snapshot.tick = 2
            holder.position = (5, 8)
            self.assertFalse(tracker.available(kind, snapshot, target))  # Cooldown even after progress.
            snapshot.tick = 5
            self.assertTrue(tracker.available(kind, snapshot, target))
            holder.position = (5, 2)
            self.assertFalse(tracker.available(kind, snapshot, target))
            self.assertTrue(tracker.available(kind, snapshot, (9, 6)))
            snapshot.sightings = (SimpleNamespace(enemy_id=3),)
            self.assertTrue(tracker.available(kind, snapshot, target))
            snapshot.sightings = ()
            snapshot.tick = 1 + SUPPORT_REFRESH_TICKS[kind]
            self.assertTrue(tracker.available(kind, snapshot, target))
            snapshot.effects = (SimpleNamespace(kind=kind, phase="active"),)
            self.assertFalse(tracker.available(kind, snapshot, target))
            snapshot.effects = ()
            snapshot.tick = 1

    def test_repeat_guard_is_shared_by_teammates_and_reset_each_round(self):
        snapshot = SimpleNamespace(tick=1, allies=(), spike_dropped=None, effects=(), sightings=())
        tracker = SupportUseTracker()
        tracker.record(snapshot, {"ability": "FLASH", "target": (5, 6)})
        self.assertFalse(tracker.available("FLASH", snapshot, (5, 6)))
        self.assertTrue(tracker.available("SMOKE", snapshot, (5, 6)))
        self.assertTrue(SupportUseTracker().available("FLASH", snapshot, (5, 6)))


if __name__ == "__main__":
    unittest.main()
