import dataclasses
from types import SimpleNamespace
import unittest

import numpy as np

from abilities_los import AbilityLosMixin
from coach_v1.common.constants import FIXED_ROSTER
from coach_v1.common.types import Facing, Side
from coach_v1.perception.team_perception import (
    PerceptionInputError,
    SightingSource,
    TeamPerceptionBuilder,
)


class FakeCharacter:
    def __init__(
        self,
        name,
        team,
        pos,
        *,
        facing="E",
        alive=True,
        hp=100,
        iq=200,
        blind=0,
        reveal=0,
        has_spike=False,
    ):
        self.name = name
        self.team = team
        self.pos = list(pos)
        self.facing = facing
        self.is_alive = alive
        self.hp = hp
        self.effective_iq = iq
        self.blind_remaining = blind
        self.reveal_remaining = reveal
        self.has_spike = has_spike
        self.smoke_charges = 1
        self.recon_charges = 1
        self.flash_charges = 1


class FakeGame(AbilityLosMixin):
    def __init__(self, grid, chars, *, smoke_cells=()):
        self.grid = np.asarray(grid, dtype=np.int8)
        self.chars = list(chars)
        self.smokes = (
            [{"cells": set(smoke_cells), "remaining_ticks": 5}]
            if smoke_cells
            else []
        )
        self.current_round = 2
        self.battle_tick = 11
        self.defender_setup_phase = SimpleNamespace(active=False, ticks_remaining=0)
        self.spike_pos = None
        self.is_planted = False
        self.planted_pos = None

def fixed_team(
    team="A",
    *,
    first_pos=(3, 1),
    first_facing="E",
    first_blind=0,
    all_alive=False,
):
    positions = [first_pos, (1, 3), (1, 1), (5, 1), (5, 5)]
    facings = [first_facing, "S", "E", "N", "W"]
    result = []
    for slot, roster_slot in enumerate(FIXED_ROSTER):
        result.append(
            FakeCharacter(
                roster_slot.character_name,
                team,
                positions[slot],
                facing=facings[slot],
                alive=all_alive or slot == 0,
                iq=200 - slot * 10,
                blind=first_blind if slot == 0 else 0,
            )
        )
    return result


def floor_grid():
    return np.zeros((7, 7), dtype=np.int8)


class CoachV1Task03TeamPerceptionTest(unittest.TestCase):
    def setUp(self):
        self.builder = TeamPerceptionBuilder()

    def test_wall_blocks_enemy_and_clear_mask(self):
        grid = floor_grid()
        grid[3, 3] = 1
        enemy = FakeCharacter("wall_enemy", "D", (3, 5), facing="W")
        snapshot = self.builder.build(
            game=FakeGame(grid, fixed_team() + [enemy]), side=Side.ATTACKER
        )

        self.assertIsNone(snapshot.sighting_for("wall_enemy"))
        self.assertFalse(snapshot.currently_visible[3][5])
        self.assertFalse(snapshot.currently_visible[3][3])

    def test_smoke_blocks_enemy_and_smoke_cell_is_not_clear(self):
        grid = floor_grid()
        enemy = FakeCharacter("smoke_enemy", "D", (3, 5), facing="W")
        snapshot = self.builder.build(
            game=FakeGame(
                grid, fixed_team() + [enemy], smoke_cells={(3, 3)}
            ),
            side="attacker",
        )

        self.assertIsNone(snapshot.sighting_for("smoke_enemy"))
        self.assertFalse(snapshot.currently_visible[3][5])
        self.assertFalse(snapshot.currently_visible[3][3])
        self.assertEqual(((3, 3),), snapshot.smoke_cells)

    def test_rear_cells_are_not_cleared_and_fov_boundary_is_included(self):
        snapshot = self.builder.build(
            game=FakeGame(floor_grid(), fixed_team()), side="attacker"
        )

        self.assertFalse(snapshot.currently_visible[3][0])
        self.assertTrue(snapshot.currently_visible[3][1])
        self.assertTrue(snapshot.currently_visible[0][1])
        self.assertTrue(snapshot.currently_visible[3][6])

    def test_best_iq_qualified_viewer_produces_one_team_shared_sighting(self):
        allies = fixed_team(all_alive=True)
        allies[0].effective_iq = 100
        allies[1].effective_iq = 200
        enemy = FakeCharacter("shared_enemy", "D", (3, 3), facing="W")
        snapshot = self.builder.build(
            game=FakeGame(floor_grid(), allies + [enemy]), side="attacker"
        )

        self.assertEqual(1, len(snapshot.sightings))
        self.assertEqual((3, 3), snapshot.sightings[0].reported_position)
        self.assertEqual(1, snapshot.sightings[0].viewer_slot)
        self.assertEqual(SightingSource.NORMAL, snapshot.sightings[0].source)

    def test_equal_iq_sighting_uses_lower_roster_slot(self):
        allies = fixed_team(all_alive=True)
        allies[0].effective_iq = 200
        allies[1].effective_iq = 200
        enemy = FakeCharacter("tie_enemy", "D", (3, 3), facing="W")
        snapshot = self.builder.build(
            game=FakeGame(floor_grid(), allies + [enemy]), side="attacker"
        )

        self.assertEqual(0, snapshot.sighting_for("tie_enemy").viewer_slot)

    def test_hidden_enemy_position_changes_do_not_change_actor_snapshot(self):
        class RejectHiddenPositionReporting:
            def _blur_pos(self, *args, **kwargs):
                raise AssertionError("IQ reporting must not run for an unseen enemy")

        builder = TeamPerceptionBuilder(iq_engine=RejectHiddenPositionReporting())
        enemy_a = FakeCharacter("hidden_enemy", "D", (3, 0), facing="E")
        enemy_b = FakeCharacter("hidden_enemy", "D", (0, 0), facing="S")
        first = builder.build(
            game=FakeGame(floor_grid(), fixed_team() + [enemy_a]),
            side="attacker",
        )
        second = builder.build(
            game=FakeGame(floor_grid(), fixed_team() + [enemy_b]),
            side="attacker",
        )

        self.assertEqual(first, second)
        self.assertEqual((), first.sightings)
        self.assertEqual(("hidden_enemy",), tuple(e.enemy_id for e in first.enemies))
        self.assertFalse(hasattr(first.enemies[0], "position"))
        self.assertFalse(hasattr(first.enemies[0], "hp"))

    def test_blind_viewer_generates_no_normal_vision_but_recon_is_shared(self):
        grid = floor_grid()
        grid[3, 3] = 1
        allies = fixed_team(first_blind=2)
        allies[2].effective_iq = 200
        enemy = FakeCharacter(
            "recon_enemy", "D", (3, 5), facing="W", reveal=2
        )
        snapshot = self.builder.build(
            game=FakeGame(grid, allies + [enemy], smoke_cells={(3, 4)}),
            side="attacker",
        )

        self.assertFalse(any(any(row) for row in snapshot.currently_visible))
        sighting = snapshot.sighting_for("recon_enemy")
        self.assertIsNotNone(sighting)
        self.assertEqual((3, 5), sighting.reported_position)
        self.assertEqual(2, sighting.viewer_slot)
        self.assertEqual(SightingSource.RECON, sighting.source)

    def test_viewer_count_integrates_only_alive_unblinded_teammates(self):
        allies = fixed_team(all_alive=True)
        allies[2].blind_remaining = 1
        allies[3].is_alive = False
        allies[4].is_alive = False
        snapshot = self.builder.build(
            game=FakeGame(floor_grid(), allies), side="attacker"
        )

        self.assertEqual(2, snapshot.visible_viewer_count[3][3])
        self.assertEqual(
            snapshot.currently_visible[3][3],
            snapshot.visible_viewer_count[3][3] > 0,
        )

    def test_spike_info_exposes_own_carrier_but_never_enemy_carrier(self):
        attackers = fixed_team(team="A")
        attackers[3].has_spike = True
        attacker_snapshot = self.builder.build(
            game=FakeGame(floor_grid(), attackers), side="attacker"
        )
        self.assertEqual(3, attacker_snapshot.spike.own_carrier_slot)

        defenders = fixed_team(team="D")
        enemy_carrier = FakeCharacter(
            "enemy_carrier", "A", (3, 0), has_spike=True
        )
        defender_snapshot = self.builder.build(
            game=FakeGame(floor_grid(), defenders + [enemy_carrier]),
            side="defender",
        )
        self.assertIsNone(defender_snapshot.spike.own_carrier_slot)
        self.assertFalse(hasattr(defender_snapshot.enemies[0], "has_spike"))

    def test_dropped_and_planted_spike_positions_are_copied(self):
        game = FakeGame(floor_grid(), fixed_team())
        game.spike_pos = [2, 2]
        dropped = self.builder.build(game=game, side="attacker")
        game.spike_pos[0] = 6
        self.assertEqual((2, 2), dropped.spike.dropped_position)

        game.spike_pos = None
        game.is_planted = True
        game.planted_pos = [4, 4]
        planted = self.builder.build(game=game, side="attacker")
        self.assertEqual((4, 4), planted.spike.planted_position)

    def test_snapshot_is_frozen_and_does_not_retain_live_objects(self):
        allies = fixed_team()
        snapshot = self.builder.build(
            game=FakeGame(floor_grid(), allies), side="attacker"
        )
        allies[0].pos[0] = 0

        self.assertEqual((3, 1), snapshot.allies[0].position)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.side = Side.DEFENDER
        self.assertFalse(any(ally is raw for ally in snapshot.allies for raw in allies))

    def test_tick_key_uses_setup_countdown_instead_of_battle_tick(self):
        game = FakeGame(floor_grid(), fixed_team(team="D"))
        game.battle_tick = 99
        game.defender_setup_phase.active = True
        game.defender_setup_phase.ticks_remaining = 7
        snapshot = self.builder.build(game=game, side="defender")

        self.assertEqual((2, "defender_setup", 7), dataclasses.astuple(snapshot.tick))

    def test_fixed_roster_contract_is_enforced(self):
        allies = fixed_team()
        allies[0].name = "not_gorimaru"
        with self.assertRaisesRegex(PerceptionInputError, "fixed Gorigons roster"):
            self.builder.build(
                game=FakeGame(floor_grid(), allies), side="attacker"
            )


if __name__ == "__main__":
    unittest.main()
