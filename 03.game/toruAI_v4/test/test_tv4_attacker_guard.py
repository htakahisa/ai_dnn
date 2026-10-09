"""Synthetic unit checks only; never run collection or live training here."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import deque
from dataclasses import replace
import copy
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from team_ai import DualRoleTeamAI
from public_effects import DisplayEffect
from frc_v1.perception import FrcPerceptionBuilder, Sighting
from frc_v1.actions import build_masks, validate_action
from grid_paths import distance_map
from concon_v1.co1_retake_cases import save_case, load_case, case_metadata
from toruAI_v4.test.test_tv4_attacker_analysis import world
from toruAI_v4.tv4_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel
from toruAI_v4.tv4_learn_attacker_plant import PlantDQN
from toruAI_v4.tv4_learn_attacker_guard import (
    GuardDQN, GuardEncoder, OBS_DIM, ACTION_DIM, DISABLED_DEFUSE_ACTION,
    guard_schema, learn_guard, load_guard, crossfire_score,
    guard_fire_line, guard_defuse_cells,
)
from toruAI_v4.tv4_attacker_guard_controller import ToruV4AttackerGuardController, ToruV4AttackerPlantGuardController
from toruAI_v4.tv4_guard_runtime import CASE_FORMAT, summarize_guard, guard_best_rank
from toruAI_v4.tv4_collect_attacker_guard import valid_guard_case, validate_output
from toruAI_v4.tv4_train_attacker_guard import save_latest, consider_best, load_dataset


def planted_world():
    game = world()
    cells = game.sites["L"]
    for char, pos in zip(game.chars[:5], cells[:5]):
        char.pos = list(pos)
        char.has_spike = False
    game.is_planted, game.planted_pos = True, cells[0]
    game.headless, game.round_over, game.match_over, game.is_defused = True, False, False, False
    game.battle_tick, game.detonate_timer = 20, 54
    game.attacker_wins = game.defender_wins = 0
    return game


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self, game):
        controller = ToruV4AttackerGuardController(game, GuardDQN())
        controller.set_game(game)
        controller.prepare_team_tick()
        return controller

    def test_hidden_enemy_positions_do_not_change_public_guard_actions(self):
        game = planted_world()
        first = self.controller(game)
        first.training, first.teacher_probability = True, 1.
        first.cache = None
        first.prepare_team_tick()
        actions = copy.deepcopy(first.actions)
        observations = {n: x.observation.copy() for n, x in first.inputs.items()}
        for i, enemy in enumerate(game.chars[5:]):
            enemy.pos = [2, 6 + i]
        second = ToruV4AttackerGuardController(game, first.policy)
        second.training, second.teacher_probability = True, 1.
        second.set_game(game)
        second.prepare_team_tick()
        self.assertEqual(actions, second.actions)
        self.assertEqual(first.goals, second.goals)
        for n, observation in observations.items():
            np.testing.assert_array_equal(observation, second.inputs[n].observation)
            self.assertEqual(len(observation), OBS_DIM)

    def test_all_candidate_actions_legal_and_no_defuse(self):
        game = planted_world()
        controller = self.controller(game)
        masks = build_masks(controller.snapshot)
        self.assertEqual(len(set(controller.goals.values())), 5)
        for _, (_, inputs, ally) in controller.plans.items():
            self.assertFalse(inputs.mask[DISABLED_DEFUSE_ACTION])
            self.assertTrue(inputs.mask[inputs.teacher])
            for i in np.flatnonzero(inputs.mask):
                validate_action(controller.snapshot, masks, ally.slot, inputs.actions[i])

    def test_blind_counterflash_ignores_existing_enemy_flash_but_obeys_disable_mask(self):
        game = planted_world()
        snapshot = FrcPerceptionBuilder("A").build(game)
        allies = list(snapshot.allies)
        ally = replace(allies[0], ability_name="FLASH", charges=1, blind=5)
        allies[0] = ally
        target = next(p for p in game.neighbors(ally.position) if p != ally.position)
        snapshot = replace(snapshot, allies=tuple(allies),
            sightings=(Sighting(0, target, "normal"),),
            effects=(DisplayEffect(1, "FLASH", "active", target),))
        encoder = GuardEncoder(game)
        tracks = {0: (target, snapshot.tick, (0., 0.))}
        inputs = encoder.encode(snapshot, ally, ally.position, tracks, build_masks(snapshot), 20, 54, 5, 5, 1)
        self.assertEqual(inputs.actions[inputs.teacher].kind, "ABILITY")
        disabled = replace(ally, movement_disabled=2)
        allies[0] = disabled
        blocked_snapshot = replace(snapshot, allies=tuple(allies))
        blocked = encoder.encode(blocked_snapshot, disabled, disabled.position, tracks, build_masks(blocked_snapshot), 20, 54, 5, 5, 1)
        self.assertEqual(blocked.actions[blocked.teacher].kind, "STAY")

    def test_crossfire_favors_opposite_angles_over_same_angle(self):
        game = planted_world()
        target = (22, 20)
        teammate = (22, 17)
        opposite = crossfire_score(game, (22, 23), (teammate,), (target,))
        same = crossfire_score(game, (22, 18), (teammate,), (target,))
        self.assertGreater(opposite, same)

    def smoke_world(self):
        game = planted_world()
        zone = guard_defuse_cells(game, game.planted_pos)
        smoke = {p for p, _ in game.local(game.planted_pos, 4)}
        game._smoke_cells = lambda: smoke
        game.active_defuser_name = game.chars[5].name
        for char in game.chars[1:5]:
            char.is_alive = False
        game.chars[0].pos = list(next(p for p, d in game.local(game.planted_pos, 4) if d == 4))
        return game, zone

    def test_smoke_defuse_overrides_camping_policy_and_reaches_firing_range(self):
        game, zone = self.smoke_world()
        model = GuardDQN()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.net[-1].bias[0] = 100.  # Strong preference to stay forever.
            model.net[-1].bias[40] = 90.  # Then prefer casting utility.
        controller = ToruV4AttackerGuardController(game, model)
        controller.set_game(game)
        field = distance_map(game.grid, game.planted_pos)
        for _ in range(4):
            old = field[tuple(game.chars[0].pos)]
            controller.prepare_team_tick()
            chosen, inputs, ally = next(iter(controller.plans.values()))
            validate_action(controller.snapshot, build_masks(controller.snapshot), ally.slot, inputs.actions[chosen])
            self.assertTrue(inputs.defuse_pressure)
            self.assertNotIn(inputs.actions[chosen].kind, ("STAY", "ABILITY", "ULTIMATE"))
            game.chars[0].pos = list(controller.actions[ally.name][0])
            self.assertLess(field[tuple(game.chars[0].pos)], old)
            game.battle_tick += 1
        self.assertEqual(tuple(game.chars[0].pos), game.planted_pos)
        self.assertTrue(all(guard_fire_line(game, game.planted_pos, p, game._smoke_cells()) for p in zone))

    def test_smoke_prepositions_teacher_before_notification_and_respects_disable(self):
        game, _ = self.smoke_world()
        game.active_defuser_name = None
        controller = ToruV4AttackerGuardController(game, None)
        controller.set_game(game)
        controller.prepare_team_tick()
        chosen, inputs, _ = next(iter(controller.plans.values()))
        self.assertEqual(inputs.goal, game.planted_pos)
        self.assertNotIn(inputs.actions[chosen].kind, ("STAY", "ABILITY", "ULTIMATE"))
        snapshot = replace(controller.snapshot, defuse_notified=True,
                           allies=(replace(controller.snapshot.allies[0], movement_disabled=2, blind=10),)
                           + controller.snapshot.allies[1:])
        controller.sensor = SimpleNamespace(build=lambda game: snapshot)
        controller.cache = None
        controller.prepare_team_tick()
        chosen, inputs, _ = next(iter(controller.plans.values()))
        self.assertEqual(inputs.actions[chosen].kind, "STAY")

    def test_visible_nearby_enemy_does_not_cancel_smoke_defuse_approach(self):
        game, zone = self.smoke_world()
        origin = next(p for p in game.neighbors(game.planted_pos)
                      if p in zone)
        game.chars[0].pos = list(origin)
        snapshot = FrcPerceptionBuilder("A").build(game)
        # A nearby sighting used to switch off the approach override even
        # though other defuse cells were still hidden by smoke.
        nearby_enemy = next(p for p in zone if p not in (origin, game.planted_pos)
                            and guard_fire_line(game, origin, p, game._smoke_cells()))
        snapshot = replace(snapshot, sightings=(Sighting(0, nearby_enemy, "normal"),))
        model = GuardDQN()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.net[-1].bias[0] = 100.
        controller = ToruV4AttackerGuardController(game, model,
            sensor=SimpleNamespace(build=lambda game: snapshot))
        controller.set_game(game)
        controller.prepare_team_tick()
        chosen, inputs, ally = next(iter(controller.plans.values()))
        self.assertTrue(inputs.defuse_pressure)
        self.assertEqual(inputs.goal, game.planted_pos)
        self.assertEqual(tuple(controller.actions[ally.name][0]), game.planted_pos)
        self.assertNotEqual(inputs.actions[chosen].kind, "STAY")
        validate_action(snapshot, build_masks(snapshot), ally.slot, inputs.actions[chosen])

    def test_opposite_side_of_spike_requires_closing_then_aiming_at_enemy(self):
        game, _ = self.smoke_world()
        plant, origin, target = next(
            (p, (p[0], p[1] - 1), (p[0], p[1] + 1))
            for p in game.sites["L"]
            if (p[0], p[1] - 1) in set(game.neighbors(p))
            and (p[0], p[1] + 1) in set(game.neighbors(p)))
        game.planted_pos = plant
        game._smoke_cells = lambda: {plant, origin, target}
        game.chars[0].pos = list(origin)
        game.chars[5].pos = list(target)
        model = GuardDQN()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.net[-1].bias[0] = 100.
        controller = ToruV4AttackerGuardController(game, model)
        controller.set_game(game)
        controller.prepare_team_tick()
        self.assertFalse(guard_fire_line(game, origin, target, game._smoke_cells()))
        self.assertEqual(tuple(controller.actions[game.chars[0].name][0]), plant)
        game.chars[0].pos = list(plant)
        game.battle_tick += 1
        snapshot = FrcPerceptionBuilder("A").build(game)
        snapshot = replace(snapshot, sightings=(Sighting(0, target, "normal"),))
        controller.sensor = SimpleNamespace(build=lambda game: snapshot)
        controller.prepare_team_tick()
        chosen, inputs, ally = next(iter(controller.plans.values()))
        self.assertEqual(inputs.actions[chosen].kind, "STAY")
        self.assertEqual(inputs.actions[chosen].facing, "E")
        self.assertTrue(guard_fire_line(game, plant, target, game._smoke_cells()))
        validate_action(snapshot, build_masks(snapshot), ally.slot, inputs.actions[chosen])
        # Check the same positions and selected facing against real shooting,
        # rather than stopping at the controller's proposed action.
        from test.test_ultimate_system import UltimateTestGame
        from game_core import SHOOT_INTERVAL_TICKS
        engine = UltimateTestGame()
        engine.grid = game.grid.copy()
        attacker, defuser = game.chars[0], game.chars[5]
        engine.chars = [attacker, defuser]
        engine.smokes = [{"cells": {origin, plant, target}, "remaining_ticks": 10}]
        engine.is_planted, engine.planted_pos = True, plant
        engine.battle_tick = SHOOT_INTERVAL_TICKS
        defuser.defuse_timer = 1
        attacker.pos = list(origin)
        self.assertFalse(engine.check_shot_line_of_sight(attacker, defuser))
        attacker.pos = list(plant)
        attacker.facing = inputs.actions[chosen].facing
        self.assertTrue(engine.check_shot_line_of_sight(attacker, defuser))
        with patch("battle_logic.random.random", return_value=0.):
            engine._resolve_all_shots()
        self.assertTrue(any(shot["shooter"] is attacker and shot["target"] is defuser
                            for shot in engine.last_shots))

    def test_blinded_smoke_defuse_closes_then_sweeps_instead_of_counterflash(self):
        game, _ = self.smoke_world()
        game.chars[0].blind_remaining = 10
        model = GuardDQN()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.net[-1].bias[40] = 100.
        controller = ToruV4AttackerGuardController(game, model)
        controller.set_game(game)
        facings = set()
        for tick in range(8):
            snapshot = FrcPerceptionBuilder("A").build(game)
            snapshot = replace(snapshot, allies=(replace(snapshot.allies[0],
                               ability_name="FLASH", charges=1, blind=10),) + snapshot.allies[1:])
            controller.sensor = SimpleNamespace(build=lambda game: snapshot)
            controller.prepare_team_tick()
            chosen, inputs, ally = next(iter(controller.plans.values()))
            self.assertTrue(inputs.blind)
            self.assertTrue(inputs.defuse_pressure)
            self.assertEqual(inputs.goal, game.planted_pos)
            validate_action(snapshot, build_masks(snapshot), ally.slot, inputs.actions[chosen])
            if tick < 4:
                self.assertNotIn(inputs.actions[chosen].kind, ("STAY", "ABILITY", "ULTIMATE"))
            else:
                self.assertEqual(inputs.actions[chosen].kind, "STAY")
                facings.add(inputs.actions[chosen].facing)
            self.assertEqual(inputs.teacher, chosen)
            game.chars[0].pos = list(controller.actions[ally.name][0])
            game.battle_tick += 1
        self.assertEqual(tuple(game.chars[0].pos), game.planted_pos)
        self.assertGreater(len(facings), 1)

    def test_all_manual_ability_types_have_delay_demonstrations(self):
        game = planted_world()
        original = FrcPerceptionBuilder("A").build(game)
        encoder = GuardEncoder(game)
        target = next(game.neighbors(original.allies[0].position))
        for kind in ("FLASH", "SMOKE", "ASH", "RECON", "RAMP", "DANCE"):
            with self.subTest(ability=kind):
                allies = list(original.allies)
                ally = replace(allies[0], ability_name=kind, charges=1)
                allies[0] = ally
                allies[1] = replace(allies[1], hp=30.)
                current = () if kind == "RECON" else (Sighting(0, target, "normal"),)
                snapshot = replace(original, allies=tuple(allies), sightings=current)
                tracks = {0: (target, snapshot.tick, (0., 0.))}
                inputs = encoder.encode(snapshot, ally, ally.position, tracks, build_masks(snapshot),
                                        snapshot.tick - 6, 54, 5, 5, 1)
                self.assertEqual(inputs.actions[inputs.teacher].kind, "ABILITY")
                validate_action(snapshot, build_masks(snapshot), ally.slot, inputs.actions[inputs.teacher])
                if kind == "SMOKE":
                    smoke_target = inputs.actions[inputs.teacher].target
                    self.assertGreater(max(abs(smoke_target[0] - snapshot.spike_planted[0]),
                                           abs(smoke_target[1] - snapshot.spike_planted[1])), 1)
                if kind == "RAMP":
                    again = encoder.encode(snapshot, ally, ally.position, tracks, build_masks(snapshot),
                                           snapshot.tick - 6, 54, 5, 5, 1, {ally.position})
                    self.assertFalse(any(again.mask[i] for i in range(40, 48)))

    def test_plant_handoff_waits_until_next_tick_and_keeps_memory(self):
        game = planted_world()
        encoder = AttackerEncoder(game)
        analysis = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(game.names))
        combined = ToruV4AttackerPlantGuardController(game, analysis, PlantDQN(), GuardDQN())
        combined.set_game(game)
        combined.plant.cache = (1, "live", game.battle_tick, False)
        combined.plant.analysis_encoder.history.tracks = {0: ((14, 16), 19, (0., 0.))}
        before = list(game.chars[0].pos)
        combined.decide_move(game.chars[0], {})
        self.assertIsNone(combined.guard)
        self.assertEqual(game.chars[0].pos, before)
        game.battle_tick += 1
        combined.prepare_team_tick()
        self.assertIsNotNone(combined.guard)
        self.assertIn(0, combined.guard.history.tracks)
        self.assertIs(combined.plant.sensor, combined.guard.sensor)

    def test_full_case_save_load_keeps_boundary_and_controller_state(self):
        game = planted_world()
        encoder = AttackerEncoder(game)
        analysis = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(game.names))
        combined = ToruV4AttackerPlantGuardController(game, analysis, PlantDQN())
        combined.set_game(game)
        combined.mark_plant_boundary()
        defender = SimpleNamespace(game=game, rounds_seen=7)
        game.attacker_controller, game.defender_controller = combined, defender
        game.current_attacker_team_ai = DualRoleTeamAI("test A", lambda: combined, lambda: defender)
        game.current_defender_team_ai = DualRoleTeamAI("test D", lambda: combined, lambda: defender)
        game.current_attacker_team_ai._attacker_controller = combined
        game.current_defender_team_ai._defender_controller = defender
        valid_guard_case(game)
        metadata = {**case_metadata(game, "gc_v1"), "format": CASE_FORMAT}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.case.gz"
            save_case(path, game, metadata)
            restored, info = load_case(path)
            self.assertEqual(info["format"], CASE_FORMAT)
            self.assertEqual(restored.battle_tick, restored.attacker_controller.guard.start_tick)
            self.assertEqual(restored.defender_controller.rounds_seen, 7)
            self.assertEqual(restored.attacker_controller.guard.goals, combined.guard.goals)
            self.assertEqual([c.pos for c in restored.chars], [c.pos for c in game.chars])

    def test_latest_and_best_are_distinct_and_equal_or_worse_never_replaces_best(self):
        game = planted_world()
        controller = self.controller(game)
        inputs = next(iter(controller.inputs.values()))
        replay = deque([[inputs.observation.astype(np.float16), inputs.teacher, 1.,
                         np.zeros(OBS_DIM, np.float16), np.ones(ACTION_DIM, bool), 1., inputs.teacher, inputs.mask.copy()]] * 4)
        model, target = GuardDQN(), GuardDQN()
        optimizer = torch.optim.Adam(model.parameters())
        self.assertTrue(np.isfinite(learn_guard(model, target, optimizer, replay, np.random.default_rng(0), 1, 4)))
        record = {"guard_played": True, "guard_won": True, "full_round_won": True, "planted": True,
            "initial_alive": 5, "initial_enemy_alive": 5, "initial_abilities": 2,
            "remaining_abilities_alive": 1, "alive": 3, "enemy_alive": 1, "uses": 1,
            "damage": 100, "reward": 8, "reason": "exploded", "site": "L"}
        evaluation = {"metrics": summarize_guard([record]), "seeds": [42]}
        contract = {"schema": guard_schema(game), "opponent": "gc_v1", "source_hashes": {"plant": "p", "analysis": "a"}}
        with tempfile.TemporaryDirectory() as directory:
            latest, best = Path(directory) / "latest.pt", Path(directory) / "attacker_guard_best.pt"
            self.assertTrue(consider_best(best, model, contract, 1, 50, evaluation))
            original = best.read_bytes()
            save_latest(latest, model, target, optimizer, replay, np.random.default_rng(0), contract, 2, 100, None)
            self.assertFalse(consider_best(best, model, contract, 2, 100, evaluation, evaluation))
            worse = {"metrics": {**evaluation["metrics"], "guard_win_rate": 0.}, "seeds": [42]}
            self.assertFalse(consider_best(best, model, contract, 3, 150, worse, evaluation))
            self.assertEqual(original, best.read_bytes())
            self.assertEqual(torch.load(latest, weights_only=True)["completed_sets"], 2)
            self.assertEqual(load_guard(best, game, "gc_v1", contract["source_hashes"])[1]["completed_sets"], 1)
            empty = {"metrics": summarize_guard([]), "seeds": [42]}
            self.assertFalse(consider_best(best, model, contract, 4, 200, empty, evaluation))
            self.assertEqual(evaluation["metrics"]["by_site"]["R"]["win_rate"], None)

    def test_collection_never_accepts_nonplant_or_empty_enemy_case_and_protects_plant_data(self):
        game = planted_world()
        valid_guard_case(game)
        game.is_planted = False
        with self.assertRaises(ValueError):
            valid_guard_case(game)
        game.is_planted = True
        for char in game.chars[5:]:
            char.is_alive = False
        with self.assertRaises(ValueError):
            valid_guard_case(game)
        with self.assertRaises(ValueError):
            validate_output(Path(__file__).resolve().parents[1] / "data" / "attacker_plant")

    def test_dataset_identity_stays_same_for_selected_opponents_and_detects_changes(self):
        game = planted_world()
        opponents = ("gc_v1", "fnatic_v3")
        hashes = {o: {"plant": o + "p", "analysis": o + "a"} for o in opponents}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = {"format": CASE_FORMAT, "schema": guard_schema(game), "opponents": list(opponents),
                "source_hashes": hashes, "source_paths": {o: {"plant": str(directory / "best" / o / "p.pt"),
                    "analysis": str(directory / "best" / o / "a.pt")} for o in opponents}}
            (directory / "collection.json").write_text(json.dumps(manifest), encoding="utf-8")
            (directory / "collection_summary.json").write_text(json.dumps({"complete": True, "target_per_ai": 1}), encoding="utf-8")
            rows = []
            for o in opponents:
                filename = o + ".case.gz"
                (directory / filename).write_bytes(o.encode())
                rows.append({"opponent": o, "file": filename, "source_hashes": hashes[o],
                             "sha256": hashlib.sha256(o.encode()).hexdigest()})
            (directory / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            def sources(scenario, selected, plant_dir, analysis_dir):
                return {o: {"hashes": hashes[o], "generic_roster_training": True} for o in selected}
            with patch("toruAI_v4.tv4_train_attacker_guard.load_sources", side_effect=sources):
                complete_hash = load_dataset(directory, opponents, game)[2]
                selected_hash = load_dataset(directory, ("gc_v1",), game)[2]
                self.assertEqual(complete_hash, selected_hash)
                (directory / rows[0]["file"]).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    load_dataset(directory, ("gc_v1",), game)


if __name__ == "__main__":
    unittest.main()
