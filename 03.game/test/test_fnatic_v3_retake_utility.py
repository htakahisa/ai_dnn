"""Strict retake order verified against real casts, impacts and team movement."""

import unittest

import numpy as np

from fnatic_v3.controller import FnaticV3DefenderController
from fnatic_v3.positions import parse_grid
from iq_controller_adapter import IQAwareController
from iq_perception import PerceivedCharacter
from map_data import NEW_MAZE_STR
from test_fnatic_v3_retake import retake_map
from test_ultimate_system import UltimateTestGame, make_character


class FnaticRetakeUtilityTests(unittest.TestCase):
    def fixture(self, site='A', use_iq=False, mapped=True, ready=True):
        game = UltimateTestGame(12, 28)
        game.grid[0, :] = game.grid[-1, :] = 1
        game.grid[:, 0] = game.grid[:, -1] = 1
        game.grid[3:9, 2:5] = game.grid[3:9, 23:26] = 2
        game.is_planted = True
        game.planted_pos = (5, 3) if site == 'A' else (5, 24)
        col = 6 if site == 'A' else 21
        slots = ((2, col), (4, col), (6, col))
        ctrl = FnaticV3DefenderController('', retake_map(
            game.grid, slots if site == 'A' else (), slots if site == 'B' else ()) if mapped else '', engineer_map='')
        game.chars = [make_character(name, 'D', cell) for name, cell in zip(
            ('Chronicle', 'Derke', 'Leo', 'Alfajer', 'Boaster'), (*slots, (1, col), (2, col + 1)))]
        for char in game.chars:
            char.iq = char.effective_iq = 200
        enemy = make_character('Enemy', 'A', game.planted_pos)
        game.chars.append(enemy)
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        game.defender_controller = adapter
        if mapped and ready:
            ctrl.decide_move(game.chars[0], self.state(game))
            for char in game.chars[:5]:
                char.pos = list(ctrl.retake.targets[char.name])
        events = []
        original_cast = game.execute_ai_ability
        def cast(char, action):
            accepted = original_cast(char, action)
            if accepted:
                events.append(('cast', action['ability'], char.name, tuple(action['target']), game.battle_tick))
            return accepted
        game.execute_ai_ability = cast
        for kind in ('RECON', 'FLASH'):
            original = getattr(game, '_explode_' + kind.lower())
            def explode(projectile, impact=None, kind=kind, original=original):
                cell = impact or projectile['path'][min(projectile['progress'], len(projectile['path']) - 1)]
                events.append(('impact', kind, projectile['owner'], tuple(cell), game.battle_tick))
                return original(projectile, impact)
            setattr(game, '_explode_' + kind.lower(), explode)
        return game, ctrl, enemy, events

    @staticmethod
    def state(game):
        return dict(grid=game.grid, chars=game.chars, is_planted=True,
                    planted_pos=game.planted_pos, detonate_timer=game.detonate_timer)

    def tick(self, game, ctrl, events, reverse=False):
        for char in reversed(game.chars[:5]) if reverse else game.chars[:5]:
            if not char.is_alive:
                continue
            before = tuple(char.pos)
            game.move_character(char)
            if ctrl.retake.launched and tuple(char.pos) != before:
                events.append(('peek', None, char.name, tuple(char.pos), game.battle_tick))
        game._advance_flash_projectiles()
        game._advance_recon_projectiles()
        for char in game.chars:
            char.reveal_remaining = max(0, char.reveal_remaining - 1)
        game.battle_tick += 1

    def finish(self, game, ctrl, events, reverse=False):
        for _ in range(45):
            self.tick(game, ctrl, events, reverse)
            if ctrl.retake.launched:
                return
        self.fail((ctrl.retake.utility.phase, [(c.name, c.pos) for c in game.chars]))

    def assert_order(self, events, recons=2, flashes=1, smokes=1):
        indices = lambda event, kind: [i for i, row in enumerate(events) if row[0] == event and row[1] == kind]
        recon_casts, recon_hits = indices('cast', 'RECON'), indices('impact', 'RECON')
        flash_casts, flash_hits = indices('cast', 'FLASH'), indices('impact', 'FLASH')
        smoke_casts = indices('cast', 'SMOKE')
        peeks = indices('peek', None)
        self.assertEqual(len(recon_casts), recons)
        self.assertEqual(len(recon_hits), recons)
        self.assertEqual(len(flash_casts), flashes)
        self.assertEqual(len(flash_hits), flashes)
        self.assertEqual(len(smoke_casts), smokes)
        self.assertTrue(peeks)
        if recon_hits and flash_casts:
            self.assertLess(max(recon_hits), min(flash_casts))
        if flash_hits and smoke_casts:
            self.assertLess(max(flash_hits), min(smoke_casts))
        if smoke_casts:
            self.assertLess(max(smoke_casts), min(peeks))
            self.assertLess(events[max(smoke_casts)][4], events[min(peeks)][4])
        self.assertEqual(len({events[i][4] for i in peeks}), 1)

    def test_both_sites_and_decision_orders_with_iq_obey_every_barrier(self):
        for site in ('A', 'B'):
            for reverse in (False, True):
                for use_iq in (False, True):
                    with self.subTest(site=site, reverse=reverse, use_iq=use_iq):
                        game, ctrl, enemy, events = self.fixture(site, use_iq)
                        self.finish(game, ctrl, events, reverse)
                        self.assert_order(events)
                        recon = [row for row in events if row[:2] == ('cast', 'RECON')]
                        self.assertEqual(len({row[3] for row in recon}), 2)
                        self.assertTrue(all(game.grid[row[3]] == 2 for row in recon))
                        self.assertEqual(recon[1][4] - recon[0][4], 1)
                        impacts = [row[3] for row in events if row[:2] == ('impact', 'RECON')]
                        self.assertEqual(len(set(impacts)), 2)
                        flash = next(row for row in events if row[:2] == ('cast', 'FLASH'))
                        self.assertEqual(flash[3], tuple(enemy.pos))
                        self.assertEqual(game.smokes[-1]['center'], game.planted_pos)
                        self.assertEqual({row[2] for row in events if row[0] == 'peek'}, {c.name for c in game.chars[:5]})

    def test_twenty_five_waits_but_twenty_four_starts_utility_without_missing_allies(self):
        game, ctrl, _, events = self.fixture(ready=False)
        game.chars[-2].pos = [10, 13]  # Boaster is far from his gathering slot.
        game.detonate_timer = 25
        self.tick(game, ctrl, events)
        self.assertFalse(ctrl.retake.preparing)
        self.assertFalse(ctrl.retake.launched)
        game.detonate_timer = 24
        self.tick(game, ctrl, events)
        self.assertTrue(ctrl.retake.preparing)
        self.assertFalse(ctrl.retake.launched)
        self.assertEqual([row[1] for row in events if row[0] == 'cast'], ['RECON'])
        self.assertFalse(any(row[0] == 'peek' for row in events))
        self.finish(game, ctrl, events, reverse=True)
        self.assert_order(events)

    def test_empty_utility_launches_all_players_even_with_ready_ultimates_and_revealed_enemy(self):
        for mapped in (False, True):
            for reverse in (False, True):
                for use_iq in (False, True):
                    with self.subTest(mapped=mapped, reverse=reverse, use_iq=use_iq):
                        game, ctrl, enemy, events = self.fixture(mapped=mapped, use_iq=use_iq)
                        game.detonate_timer = 24
                        enemy.reveal_remaining = 15
                        for char in game.chars[:5]:
                            char.recon_charges = char.flash_charges = char.smoke_charges = 0
                            char.ultimate_points = char.ultimate_cost
                        self.tick(game, ctrl, events, reverse)
                        self.assertTrue(ctrl.retake.launched)
                        self.assertEqual({row[2] for row in events if row[0] == 'peek'},
                                         {c.name for c in game.chars[:5]})
                        self.assertFalse(any(row[0] == 'cast' for row in events))

    def test_urgent_retake_skips_remote_recon_caster_and_uses_available_flash_and_smoke(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                game, ctrl, _, events = self.fixture(ready=False)
                game.grid[1:10, 9] = 1
                ctrl.retake = type(ctrl.retake)(retake_map(game.grid, ((2, 6), (4, 6), (6, 6)), ()))
                leo = game.chars[2]
                leo.pos = [1, 13]
                game.detonate_timer = 24
                self.finish(game, ctrl, events, reverse)
                self.assert_order(events, recons=0)
                self.assertEqual(leo.recon_charges, 2)
                self.assertLess(game.battle_tick, 10)
                self.assertEqual({row[2] for row in events if row[0] == 'peek'},
                                 {c.name for c in game.chars[:5]})

    def test_defuse_deadline_uses_current_abilities_then_launches_without_flight_or_ally_wait(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                game, ctrl, _, events = self.fixture()
                self.tick(game, ctrl, events)
                self.tick(game, ctrl, events)
                self.assertTrue(game.recon_projectiles)
                game.recon_projectiles[0]['path'] *= 20
                game.detonate_timer = 1
                before = len(events)
                self.tick(game, ctrl, events, reverse)
                self.assertFalse(ctrl.retake.launched)
                self.assertEqual({row[1] for row in events[before:] if row[0] == 'cast'},
                                 {'RECON', 'FLASH', 'SMOKE'})
                self.tick(game, ctrl, events, reverse)
                self.assertTrue(ctrl.retake.launched)
                self.assertTrue(game.recon_projectiles)
                peeks = [row for row in events if row[0] == 'peek']
                self.assertEqual({row[2] for row in peeks}, {c.name for c in game.chars[:5]})
                self.assertEqual(len({row[4] for row in peeks}), 1)

    def test_emergency_rejected_casts_do_not_hold_team_or_retry_forever(self):
        game, ctrl, _, events = self.fixture(ready=False)
        game.detonate_timer = 1
        game.execute_ai_ability = lambda char, action: False
        self.tick(game, ctrl, events)
        self.assertFalse(ctrl.retake.launched)
        self.tick(game, ctrl, events)
        self.assertTrue(ctrl.retake.launched)
        self.assertEqual(len([row for row in events if row[0] == 'peek']), 5)

    def test_waits_for_every_recon_player_and_long_flight(self):
        game, ctrl, _, events = self.fixture()
        derke = game.chars[1]
        derke.ability_name, derke.recon_charges = 'RECON', 2
        original = game.execute_ai_ability
        def slow(char, action):
            accepted = original(char, action)
            if accepted and action['ability'] == 'RECON' and char.name == 'Derke':
                projectile = game.recon_projectiles[-1]
                projectile['path'] = [projectile['path'][0]] * 24 + projectile['path']
            return accepted
        game.execute_ai_ability = slow
        self.finish(game, ctrl, events)
        self.assert_order(events, recons=4)
        self.assertEqual(len(ctrl.retake.utility.recon_targets['Derke']), 2)
        self.assertEqual(len(ctrl.retake.utility.recon_targets['Leo']), 2)

    def test_rejected_cast_retries_same_target_without_releasing_flash(self):
        game, ctrl, _, events = self.fixture()
        original = game.execute_ai_ability
        attempted = []
        def reject_once(char, action):
            if action['ability'] == 'RECON':
                attempted.append(tuple(action['target']))
                if len(attempted) == 1:
                    return False
            return original(char, action)
        game.execute_ai_ability = reject_once
        self.finish(game, ctrl, events)
        self.assert_order(events)
        self.assertEqual(attempted[0], attempted[1])

    def test_dead_caster_is_removed_but_his_flying_recon_is_still_awaited(self):
        game, ctrl, _, events = self.fixture()
        self.tick(game, ctrl, events)
        self.tick(game, ctrl, events)
        leo = game.chars[2]
        self.assertEqual(leo.recon_charges, 1)
        leo.is_alive = False
        self.finish(game, ctrl, events)
        self.assert_order(events, recons=1)
        self.assertTrue(all(row[2] != 'Leo' for row in events if row[0] == 'peek'))

    def test_no_recon_and_no_flash_skip_empty_stages_and_smoke_still_precedes_peek(self):
        game, ctrl, _, events = self.fixture()
        game.chars[0].flash_charges = game.chars[2].recon_charges = 0
        self.finish(game, ctrl, events)
        self.assert_order(events, recons=0, flashes=0)

    def test_unmapped_site_uses_same_sequence_and_single_remaining_recon_is_not_duplicated(self):
        game, ctrl, _, events = self.fixture(mapped=False)
        game.chars[2].recon_charges = 1
        self.finish(game, ctrl, events)
        self.assert_order(events, recons=1)
        self.assertEqual(ctrl.retake.selected, ())

    def test_flash_aims_at_perceived_recon_location_and_ready_ultimates_cannot_override(self):
        game, ctrl, enemy, events = self.fixture()
        for char in game.chars[:5]:
            char.ultimate_points = char.ultimate_cost
        self.tick(game, ctrl, events)
        self.tick(game, ctrl, events)
        self.assertTrue(ctrl.retake.preparing)
        state = self.state(game)
        state['chars'] = [*game.chars[:5], PerceivedCharacter(enemy, pos=[7, 4], reveal_remaining=15)]
        game.chars[2].recon_charges = 0
        game.recon_projectiles.clear()
        action = ctrl.decide_move(game.chars[0], state)
        self.assertEqual(action[1], {'ability': 'FLASH', 'target': [7, 4]})
        self.assertEqual(game.tunnel_bursts, [])
        self.assertEqual(getattr(game, 'neon_bursts', []), [])

    def test_reset_discards_pending_casts_targets_and_phase(self):
        game, ctrl, _, events = self.fixture()
        self.tick(game, ctrl, events)
        self.tick(game, ctrl, events)
        self.assertTrue(ctrl.retake.preparing)
        self.assertTrue(ctrl.retake.utility.pending)
        ctrl.reset_round()
        self.assertFalse(ctrl.retake.preparing)
        self.assertFalse(ctrl.retake.launched)
        self.assertEqual(ctrl.retake.utility.pending, {})
        self.assertEqual(ctrl.retake.utility.recon_targets, {})
        self.assertIsNone(ctrl.retake.utility.emergency_tick)
        self.assertFalse(ctrl.retake.urgent)

    def test_default_map_assigns_throwers_near_firing_positions_and_finishes_within_25_ticks(self):
        for site in ('A', 'B'):
            with self.subTest(site=site):
                game, _, enemy, events = self.fixture(mapped=False)
                game.grid = parse_grid(NEW_MAZE_STR)
                game.height, game.width = game.grid.shape
                plants = [tuple(p) for p in zip(*np.where(game.grid == 2))
                          if (p[1] < game.width // 2) == (site == 'A')]
                game.planted_pos = plants[0]
                enemy.pos = list(game.planted_pos)
                for i, char in enumerate(game.chars[:5]):
                    char.pos = [1, 17 + i]
                ctrl = FnaticV3DefenderController(position_map='', engineer_map='')
                adapter = IQAwareController(ctrl)
                adapter.set_game(game)
                game.defender_controller = adapter
                ctrl.decide_move(game.chars[0], self.state(game))
                for char in game.chars[:5]:
                    char.pos = list(ctrl.retake.targets[char.name])
                game.detonate_timer = 24
                for elapsed in range(24):
                    self.tick(game, ctrl, events)
                    game.detonate_timer -= 1
                    if ctrl.retake.launched:
                        break
                self.assertTrue(ctrl.retake.launched)
                self.assertLess(elapsed + 1, 24)
                recons = sum(row[:2] == ('cast', 'RECON') for row in events)
                flashes = sum(row[:2] == ('cast', 'FLASH') for row in events)
                self.assert_order(events, recons=recons, flashes=flashes)
                self.assertTrue(all(game.grid[p] == 2 for p in ctrl.retake.utility.recon_targets.get('Leo', ())))

    def test_recon_finds_no_survivors_flash_still_prepares_site_before_smoke(self):
        game, ctrl, enemy, events = self.fixture()
        enemy.is_alive = False
        self.finish(game, ctrl, events)
        self.assert_order(events)
        self.assertEqual(ctrl.retake.utility.revealed, {})
        self.assertTrue(all(game.grid[p] == 2 for p in ctrl.retake.utility.flash_targets))


if __name__ == '__main__':
    unittest.main()
