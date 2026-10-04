"""FRC information boundary, real role semantics and joint tick execution."""

import unittest

import numpy as np

from frc_v1 import ROSTER
from frc_v1.actions import FrcAction, TeamDecision, validate_action, to_game_action, KINDS
from frc_v1.controller import FrcController
from frc_v1.memory import FrcMemory
from frc_v1.observation import FrcObservationEncoder, metadata, validate_metadata, GRID_FIELDS
from frc_v1.perception import FrcPerceptionBuilder
from public_effects import PublicEffectReader, displayed_projectile_path
from test_ultimate_system import UltimateTestGame, make_character


def make_game():
    game = UltimateTestGame(height=12, width=20)
    game.battle_tick = 0
    game.chars = [make_character(name, "A", (9, 2 + i)) for i, name in enumerate(ROSTER)]
    game.chars.append(make_character("Derke", "D", (1, 18)))
    game.chars[0].has_spike = True
    game.chars[0].ultimate_points = 4
    game.chars[4].ultimate_points = 5
    for c in game.chars:
        c.facing = "S"
    game.grid[3, 3] = 2
    return game


def observe(game):
    sensor = FrcPerceptionBuilder("A")
    snapshot = sensor.build(game)
    belief = FrcMemory().update(snapshot)
    return snapshot, belief, FrcObservationEncoder().encode(snapshot, belief)


class RecordingActor:
    def __init__(self, actions=None):
        self.calls = []
        self.actions = actions

    def act(self, observation, snapshot, belief):
        self.calls.append((observation, snapshot, belief))
        return TeamDecision(self.actions or tuple(FrcAction(facing=a.facing) for a in snapshot.allies))


class FrcInformationTests(unittest.TestCase):
    def test_known_ally_occupancy_blocks_moves(self):
        game = make_game()
        snapshot, _, observation = observe(game)
        self.assertFalse(observation.masks.kind[3, 2])  # Jean cannot move east into Arlecchino.
        self.assertFalse(observation.masks.kind[4, 4])  # Arlecchino cannot move west into Jean.
        self.assertTrue(observation.masks.kind[4, 2])
        with self.assertRaises(ValueError):
            validate_action(snapshot, observation.masks, 3, FrcAction("E", "S"))

    def test_hidden_enemy_state_and_occupancy_do_not_change_actor_or_masks(self):
        game = make_game()
        for ally in game.chars[:5]:
            ally.blind_remaining = 10
        before = observe(game)[2]
        enemy = game.chars[-1]
        enemy.pos = [9, 1]  # occupies a possible MOVE_W destination, but unseen.
        enemy.hp = 3
        enemy.ultimate_points = enemy.ultimate_cost
        enemy.flash_charges = 9
        enemy.has_spike = True
        after = observe(game)[2]
        for attr in ("grid", "vector", "tokens", "token_mask"):
            np.testing.assert_array_equal(getattr(before, attr), getattr(after, attr))
        np.testing.assert_array_equal(before.masks.kind, after.masks.kind)
        np.testing.assert_array_equal(before.masks.target, after.masks.target)

    def test_future_flight_path_owner_and_countdown_are_not_exposed(self):
        game = make_game()
        raw = {"path": [(3, 4), (3, 5), (3, 6), (3, 7)], "progress": 1,
               "team": "D", "owner": "Derke", "ticks_alive": 1}
        game.flash_projectiles = [raw]
        before = observe(game)[2]
        raw.update(path=[(3, 4), (3, 5), (4, 5), (5, 5)], owner="hidden", team="A", ticks_alive=4)
        after = observe(game)[2]
        np.testing.assert_array_equal(before.tokens, after.tokens)
        np.testing.assert_array_equal(before.grid, after.grid)
        effect = PublicEffectReader().read(game)[0]
        self.assertEqual(((3, 4), (3, 5)), effect.trail)
        self.assertFalse(hasattr(effect, "owner"))
        self.assertFalse(hasattr(effect, "remaining_ticks"))

    def test_warning_countdown_hidden_but_shape_and_phase_observable(self):
        game = make_game()
        raw = {"cells": {(3, 3), (3, 4)}, "pos": (3, 3), "phase": "warning", "remaining_ticks": 8}
        game.neon_bursts = [raw]
        before = observe(game)[2]
        raw["remaining_ticks"] = 1
        after = observe(game)[2]
        np.testing.assert_array_equal(before.tokens, after.tokens)
        raw["phase"] = "active"
        active = observe(game)[2]
        self.assertFalse(np.array_equal(before.grid, active.grid))

    def test_only_engine_reveal_state_discloses_enemy_position(self):
        game = make_game()
        for c in game.chars[:5]:
            c.blind_remaining = 10
        self.assertFalse(observe(game)[0].sightings)
        game.chars[-1].los_revealed = True
        self.assertEqual("normal", observe(game)[0].sightings[0].source)
        game.chars[-1].los_revealed = False
        game.chars[-1].reveal_remaining = 3
        self.assertEqual("reveal", observe(game)[0].sightings[0].source)

    def test_display_geometry_shared_and_snapshot_is_copied(self):
        game = make_game()
        raw = {"path": [(3, 4), (3, 5), (3, 6)], "progress": 1}
        game.ash_projectiles = [raw]
        snapshot = observe(game)[0]
        self.assertEqual(displayed_projectile_path(raw), snapshot.effects[0].trail)
        raw["path"][0] = (0, 0)
        game.chars[0].hp = 5
        game.grid[0, 0] = 1
        self.assertEqual((3, 4), snapshot.effects[0].trail[0])
        self.assertEqual(100, snapshot.allies[0].hp)
        self.assertEqual(0, snapshot.grid[0][0])

    def test_history_does_not_follow_unseen_enemy_and_resets_each_round(self):
        game = make_game()
        sensor, memory = FrcPerceptionBuilder("A"), FrcMemory()
        game.chars[-1].reveal_remaining = 1
        memory.update(sensor.build(game))
        game.battle_tick = 1
        game.chars[-1].reveal_remaining = 0
        game.chars[-1].pos = [2, 17]
        belief = memory.update(sensor.build(game))
        self.assertEqual((1, 18), belief.last_seen[0][1])
        game.current_round = 2
        game.battle_tick = 0
        self.assertFalse(memory.update(sensor.build(game)).last_seen)

    def test_effect_birth_unknown_on_join_and_token_overflow_keeps_grid(self):
        game = make_game()
        game.balemoon_warnings = [{"pos": (i // 20, i % 20), "cells": {(i // 20, i % 20)},
                                   "remaining_ticks": i} for i in range(70)]
        snapshot, belief, observation = observe(game)
        self.assertFalse(any(e.start_known for e in belief.effects))
        self.assertTrue(all(e.predicted_remaining is None for e in belief.effects))
        self.assertEqual(6, observation.overflow)
        self.assertEqual(70, observation.grid[GRID_FIELDS.index("balemoon_warning")].sum())
        self.assertEqual(64, observation.token_mask.sum())

    def test_checkpoint_contract_rejects_map_side_and_constants_changes(self):
        snapshot = observe(make_game())[0]
        saved = metadata(snapshot.grid, "A")
        validate_metadata(saved, snapshot.grid, "A")
        with self.assertRaises(ValueError):
            validate_metadata(saved, snapshot.grid, "D")
        saved["ability_constants"]["DANCE_HEAL_HP"] = 99
        with self.assertRaises(ValueError):
            validate_metadata(saved, snapshot.grid, "A")


class FrcExecutionTests(unittest.TestCase):
    def test_replacement_members_use_actual_flash_ramp_and_neon(self):
        game = make_game()
        names = ("Derke", "Boaster", "Leo", "Alfajer", "Chronicle")
        game.chars = [make_character(name, "A", (9, 2 + i)) for i, name in enumerate(names)] + game.chars[-1:]
        snapshot, _, observation = observe(game)
        runtime_names = tuple(a.name for a in snapshot.allies)
        self.assertEqual(set(runtime_names), set(names))
        for ally in snapshot.allies:
            char = next(c for c in game.chars if c.name == ally.name)
            self.assertEqual(ally.ability_name, char.ability_name)
            self.assertEqual(ally.charges, getattr(char, char.ability_name.lower() + "_charges", 0))
            if ally.ability_name == "FLASH":
                action = FrcAction("ABILITY", ally.facing, target=(8, ally.position[1]))
                validate_action(snapshot, observation.masks, ally.slot, action)
                self.assertTrue(game.execute_ai_ability(char, to_game_action(snapshot, ally.slot, action, runtime_names)[1]))
            elif ally.ability_name == "RAMP":
                action = FrcAction("ABILITY", ally.facing, target=ally.position)
                validate_action(snapshot, observation.masks, ally.slot, action)
                self.assertTrue(game.execute_ai_ability(char, to_game_action(snapshot, ally.slot, action, runtime_names)[1]))
                char.ultimate_points = char.ultimate_cost
                current, _, current_obs = observe(game)
                action = FrcAction("ULTIMATE", ally.facing, target=(5, 10))
                validate_action(current, current_obs.masks, ally.slot, action)
                self.assertTrue(game.execute_ai_ultimate(char, to_game_action(current, ally.slot, action, runtime_names)[1]))

    def test_second_idol_heals_using_its_own_slot(self):
        game = make_game()
        game.chars[4] = make_character("Nanasaki", "A", (9, 6))
        game.chars[2].hp = 20
        snapshot, _, observation = observe(game)
        self.assertEqual(snapshot.allies[4].name, "Nanasaki")
        action = FrcAction("ABILITY", snapshot.allies[4].facing, ally_slot=2)
        validate_action(snapshot, observation.masks, 4, action)
        names = tuple(a.name for a in snapshot.allies)
        payload = to_game_action(snapshot, 4, action, names)[1]
        self.assertEqual(payload["target_name"], "Lohen")
        self.assertTrue(game.execute_ai_ability(game.chars[4], payload))
        self.assertEqual(game.chars[2].hp, 70)

    def test_model_decodes_targeted_ultimates_and_heals_outside_original_slots(self):
        from dataclasses import replace
        import torch
        from frc_v1.model import FrcActorCritic
        snapshot, belief, _ = observe(make_game())
        changed = []
        for ally in snapshot.allies:
            changed.append(replace(ally, hp=20, charges=1, points=10, cost=5,
                ability_name="DANCE" if ally.slot == 4 else "FLASH",
                ultimate_name="NEON" if ally.slot == 0 else "ESCAPE"))
        snapshot = replace(snapshot, allies=tuple(changed))
        observation = FrcObservationEncoder().encode(snapshot, belief)
        model = FrcActorCritic((len(snapshot.grid), len(snapshot.grid[0])))
        with torch.no_grad():
            for slot, head in enumerate(model.role_heads):
                head[2].weight.zero_()
                head[2].bias.zero_()
                head[2].bias[KINDS.index("ABILITY" if slot == 4 else "ULTIMATE")] = 20
            greedy = model.greedy_record(observation)
            packed, _, _, _ = model.distribution([observation], deterministic=True)
        for slot in range(5):
            self.assertGreaterEqual(greedy["target"][slot], 0)
            self.assertEqual(int(packed["target"][0, slot]), greedy["target"][slot])
            target = int(greedy["target"][slot])
            action = (FrcAction("ABILITY", "N", ally_slot=target) if slot == 4
                      else FrcAction("ULTIMATE", "N", target=divmod(target, len(snapshot.grid[0]))))
            validate_action(snapshot, observation.masks, slot, action)

    def test_three_dances_far_away_and_contract_cap(self):
        game = make_game()
        for i in range(3):
            game.chars[2].hp = 20
            snapshot, _, observation = observe(game)
            action = FrcAction("ABILITY", "S", ally_slot=2)
            validate_action(snapshot, observation.masks, 0, action)
            payload = to_game_action(snapshot, 0, action, ROSTER)[1]
            self.assertTrue(game.execute_ai_ability(game.chars[0], payload))
            self.assertEqual(70, game.chars[2].hp)
        self.assertFalse(observe(game)[2].masks.kind[0, 8])
        game.chars[0].dance_charges = 1
        game.chars[2].hp, game.chars[2].max_hp, game.chars[2].contract_max_hp_lost = 10, 40, 60
        self.assertTrue(game.execute_ai_ability(game.chars[0], {"ability": "DANCE", "target_name": "Lohen"}))
        self.assertEqual(40, game.chars[2].hp)

    def test_balemoon_warning_global_idol_death_and_automatic_serenade(self):
        game = make_game()
        owner, idol, enemy = game.chars[4], game.chars[0], game.chars[-1]
        center = tuple(owner.pos)
        self.assertTrue(game.execute_ai_ultimate(owner, {"ultimate": "BALEMOON"}))
        owner.pos = [8, 18]
        for count in (2, 1, 0):
            game._advance_balemoon_warnings()
            self.assertEqual(count, game.balemoon_warnings[0]["remaining_ticks"])
            self.assertTrue(idol.is_alive)
        game._advance_balemoon_warnings()
        self.assertFalse(idol.is_alive)
        self.assertEqual(0, idol.ultimate_points)
        self.assertEqual(3, enemy.reveal_remaining)
        self.assertEqual(center, game.destruction_areas[0]["pos"])
        snapshot = observe(game)[0]
        self.assertFalse(snapshot.allies[0].alive)
        self.assertTrue(snapshot.sightings)

    def test_furina_alive_does_not_mask_balemoon_and_hunt_has_no_active_action(self):
        masks = observe(make_game())[2].masks
        self.assertTrue(masks.kind[4, 9])
        self.assertFalse(masks.kind[2, 8])
        self.assertFalse(masks.kind[0, 9])
        self.assertFalse(masks.target[0, 0, 0])

    def test_ash_range_and_raid_direction_are_legal_masks(self):
        game = make_game()
        game.chars[2].ultimate_points = 2
        game.grid[8, 4] = 1
        snapshot, _, observation = observe(game)
        self.assertFalse(observation.masks.target[4, 0, 0])
        with self.assertRaises(ValueError):
            validate_action(snapshot, observation.masks, 2, FrcAction("ULTIMATE", "N"))

    def test_recon_launch_masks_match_game_bresenham_at_walls_and_corners(self):
        game = make_game()
        game.grid[8, 6] = 1
        game.grid[9, 6] = 1
        game.grid[10, 4] = 1
        snapshot, _, observation = observe(game)
        origin = snapshot.allies[3].position
        for r, c in np.argwhere(game.grid != 1):
            expected = len(game._projectile_path(origin, (int(r), int(c)))) > 1
            self.assertEqual(expected, observation.masks.target[3, 0, r * game.width + c])

    def test_electric_trap_blocks_movement_and_movement_ults_but_allows_dance_and_balemoon(self):
        game = make_game()
        for c in game.chars[:5]:
            c.electric_remaining = 5
            c.electric_applied_tick = None
            c.ultimate_points = c.ultimate_cost
        game.chars[2].hp = 30
        observation = observe(game)[2]
        self.assertFalse(observation.masks.kind[:, 1:5].any())
        self.assertFalse(observation.masks.kind[1, 9])
        self.assertFalse(observation.masks.kind[2, 9])
        self.assertTrue(observation.masks.kind[0, 8])
        self.assertTrue(observation.masks.kind[4, 9])

    def test_team_snapshot_frozen_before_other_side_moves_and_forced_facing_applied(self):
        game = make_game()
        actor = RecordingActor()
        controller = FrcController("A", actor=actor)
        controller.set_game(game)
        game.attacker_controller = controller
        game.chars[4].forced_facing_next_tick = "N"
        game._prepare_team_controllers_tick()
        first_snapshot = controller.snapshot
        game.chars[-1].pos = [0, 0]
        game.chars[0].pos = [10, 2]
        for c in game.chars[:5]:
            controller.decide_move(c, {})
        self.assertEqual(1, len(actor.calls))
        self.assertEqual((9, 2), first_snapshot.allies[0].position)
        self.assertEqual("N", first_snapshot.allies[4].facing)
        self.assertEqual(5, len(controller.action_log))
        game.battle_tick += 1
        controller.prepare_team_tick()
        self.assertEqual(2, len(actor.calls))

    def test_plant_action_can_be_interrupted_by_learned_retreat(self):
        game = make_game()
        game.chars[0].pos = [3, 3]
        game.chars[0].plant_timer = 2
        game.chars[0].is_planting = True
        actions = tuple(FrcAction("S" if i == 0 else "STAY", "S") for i in range(5))
        controller = FrcController("A", actor=RecordingActor(actions))
        controller.set_game(game)
        game.attacker_controller = controller
        game.move_character(game.chars[0])
        self.assertEqual([4, 3], game.chars[0].pos)
        self.assertEqual(0, game.chars[0].plant_timer)

    def test_own_effect_classification_requires_successful_resource_consumption(self):
        game = make_game()
        actions = tuple(FrcAction("ABILITY", "S", (9, 10)) if i == 4 else FrcAction(facing="S") for i in range(5))
        actor = RecordingActor(actions)
        controller = FrcController("A", actor=actor)
        controller.set_game(game)
        game.attacker_controller = controller
        game.move_character(game.chars[4])
        game.battle_tick = 1
        actor.actions = tuple(FrcAction(facing="S") for _ in range(5))
        controller.prepare_team_tick()
        self.assertEqual("own", controller.belief.effects[0].affiliation)

    def test_utility_action_applies_facing_without_overriding_forced_facing(self):
        game = make_game()
        game.chars[2].hp = 30
        actions = tuple(FrcAction("ABILITY", "W", ally_slot=2) if i == 0 else FrcAction(facing="S") for i in range(5))
        controller = FrcController("A", actor=RecordingActor(actions))
        controller.set_game(game)
        game.attacker_controller = controller
        game.move_character(game.chars[0])
        self.assertEqual("W", game.chars[0].facing)
        self.assertEqual(80, game.chars[2].hp)


if __name__ == "__main__":
    unittest.main()
