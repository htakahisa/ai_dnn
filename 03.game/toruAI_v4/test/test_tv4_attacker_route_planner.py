"""Online attack decisions from synthetic public observations, without battles."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from frc_v1.perception import FrcPerceptionBuilder, Sighting
from toruAI_v4.test.test_tv4_attacker_analysis import world
from toruAI_v4.tv4_attacker_route_planner import AdaptiveAttackPlanner, REASSESS_INTERVAL
from toruAI_v4.tv4_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel, candidate_routes
from toruAI_v4.tv4_collect_attacker_analysis import AnalysisCollectorController
from toruAI_v4.tv4_attacker_plant_controller import ToruV4AttackerPlantController
from toruAI_v4.tv4_learn_attacker_plant import PlantDQN
from toruAI_v4.tv4_train_attacker_analysis import prediction_metrics


class PublicContactModel:
    """Controlled scores derived solely from public contact side, not game state."""
    trained_rounds = 120
    def analyze(self, encoder, snapshot, observation, candidates=None):
        holder = next(a for a in snapshot.allies if a.alive and a.has_spike)
        candidates = candidates if candidates is not None else candidate_routes(encoder.scenario, holder.position, snapshot.round_timer)
        side = "L" if not snapshot.sightings or snapshot.sightings[0].position[1] >= encoder.scenario.grid.shape[1] // 2 else "R"
        rows = [{"route": r, "mode": "preserve", "score": 1. if r.site == side else 0.,
                 "information_gain": 1., "entry_pressure": float(r.site != side), "observed_empty_fraction": 0.}
                for r in candidates]
        rows.sort(key=lambda c: (-c["score"], len(c["route"].cells)))
        return {"candidates": rows, "placement": {i: {**{n: 1./len(encoder.scenario.names) for n in encoder.scenario.names}, "dead": 0.}
                for i in range(5)}, "placement_entropy": 1., "trained_rounds": 120}


class OnlineAttackTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = world()
        self.encoder = AttackerEncoder(self.game)
        self.snapshot = replace(FrcPerceptionBuilder("A").build(self.game), sightings=())
        self.model = PublicContactModel()
        self.planner = AdaptiveAttackPlanner()

    def update(self, snapshot, plan=None):
        observation = self.encoder.observe(snapshot, [])
        return self.planner.update(snapshot, observation, self.encoder, self.model,
                                   plan.route if plan else None, plan.mode if plan else None)

    def test_initial_plan_is_scout_then_contact_switches_site(self):
        first = self.update(self.snapshot)
        self.assertEqual(first.phase, "scout")
        self.assertEqual(first.route.site, "L")
        contact = Sighting(0, self.game.sites["L"][0], "normal")
        second = self.update(replace(self.snapshot, tick=3, sightings=(contact,)), first)
        self.assertTrue(second.changed)
        self.assertEqual(second.route.site, "R")
        self.assertEqual(second.reason, "public_state_changed")

    def test_same_enemy_moving_between_sites_can_trigger_more_than_two_replans(self):
        plan = self.update(self.snapshot)
        for i, occupied in enumerate(("L", "R", "L", "R", "L"), 1):
            contact = Sighting(0, self.game.sites[occupied][0], "normal")
            plan = self.update(replace(self.snapshot, tick=i*REASSESS_INTERVAL, sightings=(contact,)), plan)
            self.assertTrue(plan.changed)
            self.assertNotEqual(plan.route.site, occupied)
        self.assertEqual(len(self.planner.events), 6)

    def test_damage_triggers_immediate_review_before_periodic_interval(self):
        plan = self.update(self.snapshot)
        allies = list(self.snapshot.allies)
        allies[0] = replace(allies[0], hp=allies[0].hp-20)
        new = replace(self.snapshot, tick=1, allies=tuple(allies),
                      sightings=(Sighting(0, self.game.sites["L"][0], "normal"),))
        plan = self.update(new, plan)
        self.assertEqual(plan.route.site, "R")
        self.assertEqual(self.planner.last_tick, 1)

    def test_stall_chooses_another_first_step_and_quiet_review_does_not_oscillate(self):
        allies = tuple(replace(a, position=(18, 22)) if a.has_spike else a for a in self.snapshot.allies)
        snapshot = replace(self.snapshot, allies=allies)
        first = self.update(snapshot)
        quiet = self.update(replace(snapshot, tick=3), first)
        self.assertFalse(quiet.changed)
        stalled = self.update(replace(snapshot, tick=6), quiet)
        self.assertTrue(stalled.changed)
        self.assertNotEqual(stalled.route.cells[1], first.route.cells[1])
        self.assertEqual(stalled.reason, "stalled")

    def test_planting_commits_even_when_other_site_now_scores_better(self):
        plan = self.update(self.snapshot)
        goals = dict(plan.scout_goals)
        allies = tuple(replace(a, plant_progress=1) if a.has_spike else a for a in self.snapshot.allies)
        plan = self.update(replace(self.snapshot, tick=3, allies=allies,
                        sightings=(Sighting(0, self.game.sites["L"][0], "normal"),)), plan)
        self.assertFalse(plan.changed)
        self.assertEqual(plan.phase, "plant")
        self.assertEqual(plan.route.site, "L")
        self.assertEqual(plan.scout_goals, goals)

    def test_collector_does_not_plant_in_other_site_than_selected_route(self):
        allies = tuple(replace(a, position=self.game.sites["R"][0]) if a.has_spike else a
                       for a in self.snapshot.allies)
        snapshot = replace(self.snapshot, allies=allies)
        controller = AnalysisCollectorController(self.game, self.model, np.random.default_rng(1), exploration=0.)
        controller.set_game(self.game)
        controller.sensor.build = lambda _game: snapshot
        controller.prepare_team_tick()
        self.assertEqual(controller.route.site, "L")
        holder = next(a for a in snapshot.allies if a.has_spike)
        self.assertNotEqual(controller.actions[holder.name][1], "PLANT")

    def test_learned_enemy_beliefs_directly_change_route_ranking(self):
        model = AttackerAnalysisModel(len(self.encoder.fields), len(self.encoder.route_fields), len(self.game.names))
        routes = candidate_routes(self.game, next(a.position for a in self.snapshot.allies if a.has_spike), 100)
        left = min((r for r in routes if r.site == "L"), key=lambda r: len(r.cells))
        right = min((r for r in routes if r.site == "R"), key=lambda r: len(r.cells))
        region = int(self.game.region[left.cells[-1]])
        with torch.no_grad():
            model.placement.weight.zero_()
            model.placement.bias.fill_(-20)
            for i in range(5):
                model.placement.bias[i*(len(self.game.names)+1)+region] = 20
            model.outcome[-1].weight.zero_()
            model.outcome[-1].bias.zero_()
        analysis = model.analyze(self.encoder, self.snapshot, self.encoder.observe(self.snapshot, []), [left, right])
        pressures = {c["route"].site: c["entry_pressure"] for c in analysis["candidates"]}
        self.assertGreater(pressures["L"], pressures["R"])
        self.assertEqual(analysis["candidates"][0]["route"].site, "R")

    def test_collection_and_plant_share_contact_driven_site_switch(self):
        for role in ("collection", "plant"):
            encoder = AttackerEncoder(self.game)
            if role == "collection":
                controller = AnalysisCollectorController(self.game, self.model, np.random.default_rng(1), exploration=0.)
            else:
                controller = ToruV4AttackerPlantController(self.game, self.model, PlantDQN())
            controller.set_game(self.game)
            controller.sensor.build = lambda _game: self.snapshot
            self.game.battle_tick = 0
            controller.prepare_team_tick()
            self.assertEqual(controller.route.site, "L")
            observation = replace(self.snapshot, tick=3, sightings=(Sighting(0, self.game.sites["L"][0], "normal"),))
            controller.sensor.build = lambda _game: observation
            self.game.battle_tick = 3
            controller.prepare_team_tick()
            self.assertEqual(controller.route.site, "R")
            if role == "collection":
                self.assertEqual([f["site"] for f in controller.frames], ["L", "R"])
                self.assertGreater(controller.frames[1]["decision_id"], controller.frames[0]["decision_id"])

    def test_unopposed_online_collection_reaches_and_plants_without_fixed_route_fixture(self):
        for enemy in self.game.chars[5:]:
            enemy.is_alive = False
        for ally in self.game.chars[:5]:
            ally.flash_charges = ally.smoke_charges = ally.recon_charges = 0
        encoder = AttackerEncoder(self.game)
        model = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(self.game.names))
        with torch.no_grad():
            model.outcome[-1].weight.zero_()
            model.outcome[-1].bias.zero_()
        controller = AnalysisCollectorController(self.game, model, np.random.default_rng(1), exploration=0.)
        controller.set_game(self.game)
        progress = 0
        for tick in range(95):
            self.game.battle_tick = tick
            self.game.round_timer = 100-tick
            controller.prepare_team_tick()
            occupied = {tuple(c.pos) for c in self.game.chars[:5]}
            destinations = []
            for char in self.game.chars[:5]:
                destination, action = controller.actions[char.name]
                destination = tuple(destination)
                if destination != tuple(char.pos):
                    self.assertNotIn(destination, occupied)
                    self.assertIn(destination, set(self.game.neighbors(tuple(char.pos))))
                destinations.append(destination)
                if char.has_spike and action == "PLANT":
                    progress += 1
                    char.plant_progress = progress
            self.assertEqual(len(set(destinations)), 5)
            for char, destination in zip(self.game.chars[:5], destinations):
                char.pos = list(destination)
            if progress >= 4:
                break
        self.assertGreaterEqual(progress, 4)
        self.assertGreater(len(controller.route_planner.events), 1)

    def test_unseen_accuracy_excludes_visible_and_dead_enemy_labels(self):
        model = AttackerAnalysisModel(len(self.encoder.fields), len(self.encoder.route_fields), len(self.game.names))
        with torch.no_grad():
            model.placement.weight.zero_()
            model.placement.bias.fill_(-20)
            for i in range(5):
                model.placement.bias[i*(model.regions+1)] = 20
        x = self.encoder.observe(self.snapshot, [])
        for i in range(4):
            x[self.encoder.fields.index(f"enemy_{i}_current")] = 1.
        route = candidate_routes(self.game, next(a.position for a in self.snapshot.allies if a.has_spike), 100)[0]
        sample = {"observations": x[None], "routes": self.encoder.route_features(self.snapshot, route, "preserve")[None],
                  "placements": np.asarray([[0, 0, 0, 0, 1]]), "outcomes": np.zeros((1, 5), np.float32)}
        metrics = prediction_metrics(model, [sample], self.encoder.fields)
        self.assertAlmostEqual(metrics["placement_accuracy"], .8)
        self.assertEqual(metrics["unseen_placement_accuracy"], 0.)
        sample["placements"][0, 4] = model.regions
        self.assertIsNone(prediction_metrics(model, [sample], self.encoder.fields)["unseen_placement_accuracy"])


if __name__ == "__main__":
    unittest.main()
