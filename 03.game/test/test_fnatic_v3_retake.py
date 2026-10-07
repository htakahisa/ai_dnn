"""Retake gathering barriers through real movement, IQ and ability dispatch."""

import unittest
from types import SimpleNamespace

import numpy as np

from fnatic_v3.controller import FnaticV3DefenderController
from fnatic_v3.positions import distances, parse_grid
from fnatic_v3.retake import FnaticRetake
from iq_controller_adapter import IQAwareController
from map_data import NEW_MAZE_STR
from test_ultimate_system import UltimateTestGame, make_character


def retake_map(grid, a, b):
    rows = [['1' if cell == 1 else '0' for cell in row] for row in grid]
    for mark, cells in (('a', a), ('b', b)):
        for row, col in cells:
            rows[row][col] = mark
    return '\n'.join(''.join(row) for row in rows)


class FnaticRetakeTests(unittest.TestCase):
    def fixture(self, side='A', count=3, use_iq=False):
        grid = np.zeros((12, 28), dtype=np.int32)
        grid[3:9, 2:5] = 2
        grid[3:9, 23:26] = 2
        a = ((2, 6), (4, 6), (6, 6))
        b = ((2, 21), (4, 21), (6, 21))
        ctrl = FnaticV3DefenderController('', retake_map(grid, a, b), engineer_map='')
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        game.is_planted = True
        game.planted_pos = (5, 3) if side == 'A' else (5, 24)
        names = ('Chronicle', 'Derke', 'Leo', 'Alfajer', 'Boaster')[:count]
        game.chars = [make_character(name, 'D', (2 + 2 * i, 13)) for i, name in enumerate(names)]
        for char in game.chars:
            char.ability_name = 'NONE'
            char.iq = char.effective_iq = 200
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        game.defender_controller = adapter
        return game, ctrl, a if side == 'A' else b

    @staticmethod
    def tick(game, reverse=False):
        for char in reversed(game.chars) if reverse else game.chars:
            if char.team == 'D' and char.is_alive:
                game.move_character(char)
        game._advance_recon_projectiles()
        game._advance_flash_projectiles()
        game.battle_tick += 1

    def place_on_slots(self, game, slots):
        for char, cell in zip(game.chars, slots):
            char.pos = list(cell)

    def test_both_sites_wait_one_tick_then_launch_together_in_either_order(self):
        for site in ('A', 'B'):
            for reverse in (False, True):
                with self.subTest(site=site, reverse=reverse):
                    game, ctrl, slots = self.fixture(site)
                    self.place_on_slots(game, slots)
                    self.tick(game, reverse)
                    self.assertFalse(ctrl.retake.launched)
                    self.assertEqual([tuple(c.pos) for c in game.chars], list(slots))
                    self.tick(game, reverse)
                    self.assertTrue(ctrl.retake.launched)
                    self.assertTrue(all(tuple(c.pos) != p for c, p in zip(game.chars, slots)))

    def test_last_arrival_must_stop_before_anyone_can_charge(self):
        game, ctrl, slots = self.fixture()
        self.place_on_slots(game, slots)
        game.chars[-1].pos[1] += 1
        self.tick(game)
        self.assertEqual(tuple(game.chars[-1].pos), slots[-1])
        self.assertFalse(ctrl.retake.launched)
        self.tick(game, reverse=True)
        self.assertFalse(ctrl.retake.launched)
        self.assertEqual([tuple(c.pos) for c in game.chars], list(slots))
        self.tick(game)
        self.assertTrue(ctrl.retake.launched)

    def test_under_twenty_five_ticks_abandons_gathering_in_both_sites_and_orders(self):
        for site in ('A', 'B'):
            for reverse in (False, True):
                for remaining, launched in ((25, False), (24, True)):
                    with self.subTest(site=site, reverse=reverse, remaining=remaining):
                        game, ctrl, _ = self.fixture(site)
                        game.detonate_timer = remaining
                        lengths = distances(game.planted_pos, game.grid)
                        before = {c.name: lengths[tuple(c.pos)] for c in game.chars}
                        self.tick(game, reverse)
                        self.assertEqual(ctrl.retake.launched, launched)
                        if launched:
                            self.assertEqual(ctrl.retake.selected, ())
                            self.assertTrue(all(lengths[tuple(c.pos)] < before[c.name] for c in game.chars))

    def test_urgent_launch_ignores_missing_teammate_and_pending_stop_tick(self):
        game, ctrl, slots = self.fixture()
        self.place_on_slots(game, slots)
        game.chars[-1].pos[1] += 9
        self.tick(game)
        self.assertFalse(ctrl.retake.launched)
        before = [tuple(c.pos) for c in game.chars]
        game.detonate_timer = 24
        self.tick(game, reverse=True)
        self.assertTrue(ctrl.retake.launched)
        self.assertTrue(all(tuple(c.pos) != p for c, p in zip(game.chars, before)))

    def test_urgency_uses_real_clock_and_overrides_already_evaluated_tick(self):
        game, ctrl, _ = self.fixture()
        char = game.chars[0]
        state = dict(grid=game.grid, chars=game.chars, is_planted=True,
                     planted_pos=game.planted_pos, detonate_timer=100)
        game.detonate_timer = 25
        ctrl.decide_move(char, state)
        self.assertFalse(ctrl.retake.launched)
        phase_tick = ctrl.retake.phase_tick
        game.detonate_timer = 24
        ctrl.set_game(SimpleNamespace(real_game=game, detonate_timer=100, battle_tick=game.battle_tick))
        result = ctrl.decide_move(char, state)
        self.assertTrue(ctrl.retake.launched)
        self.assertEqual(ctrl.retake.phase_tick, phase_tick)
        lengths = distances(game.planted_pos, game.grid)
        self.assertLess(lengths[tuple(result[0])], lengths[tuple(char.pos)])

    def test_one_two_or_three_survivors_need_only_their_own_slots(self):
        for count in (1, 2, 3):
            game, ctrl, _ = self.fixture(count=count)
            self.tick(game)
            self.assertEqual(len(ctrl.retake.targets), count)
            for char in game.chars:
                char.pos = list(ctrl.retake.targets[char.name])
            self.tick(game)
            self.assertFalse(ctrl.retake.launched)
            self.tick(game)
            self.assertTrue(ctrl.retake.launched)

    def test_death_of_the_missing_player_does_not_hold_survivors_back(self):
        game, ctrl, slots = self.fixture()
        self.place_on_slots(game, slots)
        game.chars[-1].pos = [10, 13]
        self.tick(game)
        game.chars[-1].is_alive = False
        self.tick(game)
        self.assertTrue(ctrl.retake.launched)
        self.assertNotIn(game.chars[-1].name, ctrl.retake.targets)

    def test_fourth_and_fifth_survivors_wait_on_adjacent_floors_and_join_group(self):
        for count in (4, 5):
            with self.subTest(count=count):
                game, ctrl, slots = self.fixture(count=count)
                self.tick(game)
                goals = set(ctrl.retake.targets.values())
                self.assertEqual(len(goals), count)
                self.assertTrue(set(slots) <= goals)
                for extra in goals - set(slots):
                    self.assertEqual(min(abs(extra[0] - p[0]) + abs(extra[1] - p[1]) for p in slots), 1)
                for char in game.chars:
                    char.pos = list(ctrl.retake.targets[char.name])
                self.tick(game)
                self.assertFalse(ctrl.retake.launched)
                self.tick(game)
                self.assertTrue(ctrl.retake.launched)

    def test_excess_candidates_choose_only_three_and_keep_them_until_launch(self):
        for site in ('A', 'B'):
            for count in (3, 5):
                with self.subTest(site=site, count=count):
                    game, _, slots = self.fixture(side=site, count=count)
                    candidates = (*slots, (3, slots[0][1]), (5, slots[0][1]))
                    ctrl = FnaticV3DefenderController('', retake_map(
                        game.grid, candidates if site == 'A' else (), candidates if site == 'B' else ()), engineer_map='')
                    ctrl.set_game(game)
                    game.defender_controller = ctrl
                    self.tick(game)
                    selected = ctrl.retake.selected
                    self.assertEqual(len(selected), 3)
                    self.assertTrue(set(selected) <= set(candidates))
                    self.assertEqual(set(ctrl.retake.targets.values()) & set(candidates), set(selected))
                    for char in game.chars:
                        char.pos = list(ctrl.retake.targets[char.name])
                    self.tick(game, reverse=True)
                    self.assertEqual(ctrl.retake.selected, selected)
                    self.assertFalse(ctrl.retake.launched)
                    self.tick(game)
                    self.assertTrue(ctrl.retake.launched)

    def test_unreachable_excess_candidate_is_ignored(self):
        game, _, slots = self.fixture()
        game.grid[:, 8] = 1
        for i, char in enumerate(game.chars):
            char.pos = [2 + 2 * i, 7]
        ctrl = FnaticV3DefenderController('', retake_map(game.grid, (*slots, (2, 10)), ()), engineer_map='')
        ctrl.set_game(game)
        game.defender_controller = ctrl
        self.tick(game)
        self.assertEqual(set(ctrl.retake.selected), set(slots))
        self.assertNotIn((2, 10), ctrl.retake.targets.values())

    def test_group_moves_into_site_despite_visible_enemy_after_launch(self):
        game, ctrl, slots = self.fixture()
        self.place_on_slots(game, slots)
        enemy = make_character('Enemy', 'A', (5, 2))
        game.chars.append(enemy)
        self.tick(game)
        self.tick(game)
        self.assertTrue(ctrl.retake.launched)
        self.assertTrue(all(tuple(c.pos) != p for c, p in zip(game.chars[:4], slots)))

    def test_offensive_ultimates_and_smoke_recon_wait_for_group_launch(self):
        game, ctrl, slots = self.fixture(count=4)
        self.place_on_slots(game, (*slots, (1, 6)))
        for char in game.chars:
            if char.name in ('Chronicle', 'Alfajer'):
                char.ultimate_points = char.ultimate_cost
            if char.name == 'Leo':
                char.ability_name = 'RECON'
        game.smokes.append(dict(cells={(5, 3)}, owner='Enemy', team='A', remaining_ticks=10))
        self.tick(game)
        self.assertEqual(game.tunnel_bursts, [])
        self.assertEqual(getattr(game, 'neon_bursts', []), [])
        self.assertEqual(game.recon_projectiles, [])
        self.tick(game)
        self.assertTrue(ctrl.retake.preparing)
        self.assertEqual(game.tunnel_bursts, [])
        self.assertEqual(getattr(game, 'neon_bursts', []), [])
        for _ in range(25):
            self.tick(game)
            if ctrl.retake.launched:
                break
        self.assertTrue(ctrl.retake.launched)
        self.tick(game)
        self.assertIn(game.planted_pos, game.tunnel_bursts[0]['cells'])
        self.assertEqual(game.grid[game.neon_bursts[0]['pos']], 2)

    def test_monitor_can_still_launch_at_twenty_ticks_while_gathering(self):
        game, ctrl, _ = self.fixture()
        game.battle_tick = 20
        leo = next(c for c in game.chars if c.name == 'Leo')
        leo.ultimate_points = leo.ultimate_cost
        game.move_character(leo)
        self.assertTrue(ctrl.retake.gathering)
        self.assertEqual(len(game.monitor_drones), 2)

    def test_defuser_waits_for_group_even_when_already_next_to_spike(self):
        game, ctrl, slots = self.fixture('B')
        game.planted_pos = (4, 22)
        game.grid[game.planted_pos] = 2
        self.place_on_slots(game, slots)
        self.tick(game)
        self.assertTrue(all(c.defuse_timer == 0 for c in game.chars))
        self.tick(game)
        self.assertTrue(ctrl.retake.launched)
        self.assertTrue(any(c.defuse_timer > 0 for c in game.chars))

    def test_smoke_recon_resumes_once_inside_site(self):
        game, ctrl, _ = self.fixture()
        ctrl.retake.site = 'A'
        ctrl.retake.anchor = game.planted_pos
        ctrl.retake.launched = True
        leo = next(c for c in game.chars if c.name == 'Leo')
        leo.pos = [7, 3]
        leo.ability_name = 'RECON'
        game.smokes.append(dict(cells={(9, 3)}, owner='Enemy', team='A', remaining_ticks=10))
        game.move_character(leo)
        self.assertEqual(leo.recon_charges, 1)
        self.assertEqual(len(game.recon_projectiles), 1)

    def test_settled_player_swaps_slots_to_clear_a_narrow_lane(self):
        game, ctrl, _ = self.fixture(count=2)
        game.grid[:] = 1
        game.grid[4, 1:25] = 0
        game.grid[4, 2] = 2
        game.planted_pos = (4, 2)
        cells = ((4, 8), (4, 10), (4, 12), (4, 14))
        ctrl = FnaticV3DefenderController('', retake_map(game.grid, cells, ()), engineer_map='')
        ctrl.set_game(game)
        game.defender_controller = ctrl
        rear, front = game.chars
        rear.pos = [4, 7]
        front.pos = [4, 8]
        ctrl.retake.site = 'A'
        ctrl.retake.anchor = game.planted_pos
        ctrl.retake.targets = {rear.name: (4, 12), front.name: (4, 8)}
        for _ in range(10):
            self.tick(game)
            if tuple(rear.pos) == (4, 8) and tuple(front.pos) == (4, 12):
                break
        self.assertEqual(tuple(rear.pos), (4, 8))
        self.assertEqual(tuple(front.pos), (4, 12))

    def test_default_map_and_low_iq_both_sites_reach_and_hold_shared_slots(self):
        for site in ('A', 'B'):
            for count in (3, 4, 5):
                with self.subTest(site=site, count=count):
                    game, _, _ = self.fixture(count=count)
                    game.grid = parse_grid(NEW_MAZE_STR)
                    game.height, game.width = game.grid.shape
                    plants = [p for p in zip(*np.where(game.grid == 2))
                              if (p[1] < game.width // 2) == (site == 'A')]
                    game.planted_pos = plants[0]
                    for i, char in enumerate(game.chars):
                        char.pos = [1, 17 + i]
                        char.iq = char.effective_iq = 1
                    ctrl = FnaticV3DefenderController(position_map='', engineer_map='')
                    wrapper = IQAwareController(ctrl)
                    wrapper.set_game(game)
                    game.defender_controller = wrapper
                    for _ in range(160):
                        self.tick(game)
                        if ctrl.retake.launched:
                            break
                    self.assertTrue(ctrl.retake.launched, ctrl.retake.targets)
                    self.assertEqual(ctrl.retake.site, site)
                    self.assertEqual(len(ctrl.retake.targets), count)
                    self.assertEqual(len(ctrl.retake.selected), 3)

    def test_map_validation_and_empty_map_fallback(self):
        game, ctrl, _ = self.fixture()
        bad = np.zeros((2, 2), dtype=np.int32)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            ctrl.retake.validate(bad)
        game.grid[0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'walls'):
            ctrl.retake.validate(game.grid)
        for text in ('0a0\n000', '0aa\n000', '0bb\n000'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'three'):
                FnaticRetake(text)
        with self.assertRaisesRegex(ValueError, 'digits/a/b'):
            FnaticRetake('0x0\n000')
        blank = FnaticRetake('')
        blank.validate(bad)

    def test_reset_discards_gathering_and_release_decisions(self):
        game, ctrl, slots = self.fixture()
        self.place_on_slots(game, slots)
        self.tick(game)
        self.tick(game)
        self.assertTrue(ctrl.retake.launched)
        ctrl.reset_round()
        self.assertIsNone(ctrl.retake.site)
        self.assertEqual(ctrl.retake.selected, ())
        self.assertEqual(ctrl.retake.targets, {})
        self.assertEqual(ctrl.retake.stopped_at, {})
        self.assertFalse(ctrl.retake.launched)


if __name__ == '__main__':
    unittest.main()
