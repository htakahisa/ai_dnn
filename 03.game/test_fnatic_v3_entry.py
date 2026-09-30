"""Committed Derke entry and complete team planting through the real runtime."""

import unittest

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController
from fnatic_v3.formation import FRONT_SELECTION_RADIUS, UNSCREENED_PLANT_RADIUS
from fnatic_v3.positions import distances, parse_grid
from iq_controller_adapter import IQAwareController
from map_data import NEW_MAZE_STR
from test_fnatic_v3_rules import actor
from test_ultimate_system import UltimateTestGame, make_character


class FnaticDerkeEntryTests(unittest.TestCase):
    def fixture(self):
        grid = np.zeros((9, 24), dtype=np.int32)
        grid[3:6, 21:24] = 2
        ctrl = FnaticV3AttackerController(engineer_map='')
        ctrl.target = (4, 22)
        holder = actor('Leo', (4, 13))
        holder.has_spike = True
        derke = actor('Derke', (4, 16))
        chronicle = actor('Chronicle', (4, 15))
        support = actor('Support', (4, 12))
        return grid, ctrl, holder, derke, [holder, derke, chronicle, support]

    @staticmethod
    def decide(grid, ctrl, char, holder, allies, destination=None, visible=()):
        alive = [c for c in allies if c.is_alive]
        blocked = {tuple(c.pos) for c in alive if c.name != char.name}
        return ctrl.formation.result(ctrl, char, holder, alive, grid, blocked, visible,
                                     destination=destination)

    def test_front_reaches_entry_before_carrier_and_keeps_charging(self):
        grid, ctrl, holder, derke, allies = self.fixture()
        self.assertEqual(distances(ctrl.target, grid)[tuple(holder.pos)], 9)
        first = self.decide(grid, ctrl, derke, holder, allies)
        self.assertNotEqual(first[0], derke.pos)  # The old three-step slot held here.
        goal = ctrl.formation.rush_goal
        self.assertIsNotNone(goal)
        self.assertEqual(grid[goal], 2)
        self.assertNotEqual(goal, ctrl.target)
        for _ in range(15):
            before = tuple(derke.pos)
            derke.pos = self.decide(grid, ctrl, derke, holder, allies)[0]
            if tuple(derke.pos) == goal:
                break
            self.assertLess(distances(goal, grid)[tuple(derke.pos)],
                            distances(goal, grid)[before])
            self.assertEqual(ctrl.formation.rush_goal, goal)
        self.assertEqual(tuple(derke.pos), goal)
        self.assertGreater(distances(tuple(holder.pos), grid)[goal], FRONT_SELECTION_RADIUS)
        self.assertEqual(ctrl.formation.front_name, 'Derke')
        self.assertNotEqual(self.decide(grid, ctrl, holder, holder, allies)[0], holder.pos)

    def test_passage_ticket_does_not_send_committed_derke_back(self):
        grid, ctrl, holder, derke, allies = self.fixture()
        self.decide(grid, ctrl, derke, holder, allies)
        ctrl.formation.passage_waits['Support'] = ('Derke', (4, 15), (4, 14), 100)
        ctrl.formation.passage_waits['Derke'] = ('Leo', (4, 22), (4, 16), 100)
        before = tuple(derke.pos)
        result = self.decide(grid, ctrl, derke, holder, allies)
        lengths = distances(ctrl.formation.rush_goal, grid)
        self.assertLess(lengths[tuple(result[0])], lengths[before])
        self.assertNotIn('Derke', ctrl.formation.passage_waits)

    def test_visible_enemy_body_does_not_stop_entry_when_a_detour_exists(self):
        grid, ctrl, holder, derke, allies = self.fixture()
        direct = self.decide(grid, ctrl, derke, holder, allies)
        enemy = actor('Enemy', direct[0])
        enemy.team = 'D'
        action = self.decide(grid, ctrl, derke, holder, allies, visible=[enemy])
        self.assertNotEqual(action[0], derke.pos)
        self.assertNotEqual(action[0], enemy.pos)

    def test_current_rotation_setting_keeps_its_distance_boundary(self):
        for remaining in (UNSCREENED_PLANT_RADIUS, UNSCREENED_PLANT_RADIUS + 1):
            grid, ctrl, holder, _, _ = self.fixture()
            grid = np.zeros((9, 40), dtype=np.int32)
            holder.pos = [4, 28 - remaining]
            ctrl.target = (4, 28)
            ctrl.positions.plant_cells = (ctrl.target, (4, 2))
            ctrl.formation.loss_context = (holder.name, ctrl.target)
            ctrl.formation.previous_frontline = {'Derke', 'Chronicle'}
            self.decide(grid, ctrl, holder, holder, [holder])
            self.assertEqual(ctrl.target, (4, 28) if remaining == UNSCREENED_PLANT_RADIUS else (4, 2))

    def test_far_derke_is_not_recalled_to_start_entry(self):
        grid, ctrl, holder, derke, allies = self.fixture()
        derke.pos = [1, 2]
        self.decide(grid, ctrl, holder, holder, allies)
        self.assertIsNone(ctrl.formation.rush_goal)
        self.assertNotIn('Derke', ctrl.formation.front_names)

    def test_isolated_carrier_enters_a_without_waiting_for_remote_allies(self):
        grid = parse_grid(NEW_MAZE_STR)
        ctrl = FnaticV3AttackerController(engineer_map='')
        ctrl.target = (9, 3)
        holder = actor('Leo', (13, 3))
        holder.has_spike = True
        allies = [holder, actor('Derke', (2, 40)), actor('Chronicle', (2, 18)),
                  actor('Boaster', (2, 19)), actor('Alfajer', (13, 23))]
        result = self.decide(grid, ctrl, holder, holder, allies)
        self.assertEqual(ctrl.formation.front_names, ())
        self.assertEqual(ctrl.formation.mid_name, 'Alfajer')
        self.assertEqual(result[0], [12, 3])

    def test_carrier_follows_players_in_site_more_than_seven_walksteps_away(self):
        grid = parse_grid(NEW_MAZE_STR)
        ctrl = FnaticV3AttackerController(engineer_map='')
        ctrl.target = (9, 3)
        holder = actor('Leo', (13, 6))
        holder.has_spike = True
        derke = actor('Derke', (8, 3))
        chronicle = actor('Chronicle', (7, 3))
        allies = [holder, derke, chronicle, actor('Alfajer', (13, 23))]
        self.assertGreater(distances(tuple(holder.pos), grid)[tuple(chronicle.pos)],
                           FRONT_SELECTION_RADIUS)
        result = self.decide(grid, ctrl, holder, holder, allies)
        self.assertNotEqual(result[0], holder.pos)
        lengths = distances(ctrl.target, grid)
        self.assertLess(lengths[tuple(result[0])], lengths[tuple(holder.pos)])

    def test_carrier_follows_committed_derke_before_eight_step_entry_boundary(self):
        grid = parse_grid(NEW_MAZE_STR)
        ctrl = FnaticV3AttackerController(engineer_map='')
        ctrl.target = (9, 3)
        holder = actor('Leo', (15, 6))
        holder.has_spike = True
        allies = [holder, actor('Derke', (10, 3)), actor('Chronicle', (7, 3)),
                  actor('Alfajer', (13, 23))]
        lengths = distances(ctrl.target, grid)
        self.assertEqual(lengths[tuple(holder.pos)], 9)
        result = self.decide(grid, ctrl, holder, holder, allies)
        self.assertIsNotNone(ctrl.formation.rush_goal)
        self.assertLess(lengths[tuple(result[0])], lengths[tuple(holder.pos)])

    def test_carrier_follows_chronicle_in_a_site_even_outside_entry_radius(self):
        grid = parse_grid(NEW_MAZE_STR)
        ctrl = FnaticV3AttackerController(engineer_map='')
        ctrl.target = (9, 3)
        holder = actor('Leo', (9, 10))
        holder.has_spike = True
        allies = [holder, actor('Chronicle', (7, 3)), actor('Alfajer', (13, 23))]
        self.assertEqual(distances(ctrl.target, grid)[tuple(holder.pos)], 11)
        result = self.decide(grid, ctrl, holder, holder, allies)
        self.assertIsNone(ctrl.formation.rush_goal)
        self.assertEqual(result[0], [9, 9])

    def test_teammate_inside_opposite_site_does_not_trigger_entry(self):
        grid = parse_grid(NEW_MAZE_STR)
        ctrl = FnaticV3AttackerController(engineer_map='')
        ctrl.target = (9, 3)
        holder = actor('Leo', (9, 10))
        holder.has_spike = True
        allies = [holder, actor('Chronicle', (4, 40)), actor('Alfajer', (13, 23))]
        result = self.decide(grid, ctrl, holder, holder, allies)
        self.assertIsNone(ctrl.formation.entry_context)
        self.assertIsNone(ctrl.formation.rush_goal)
        self.assertEqual(result[0], [9, 9])  # The remote player is outside the regroup radius.

    def test_isolated_carrier_reaches_a_and_completes_plant_through_runtime(self):
        for use_iq in (False, True):
            with self.subTest(use_iq=use_iq):
                grid = parse_grid(NEW_MAZE_STR)
                ctrl = FnaticV3AttackerController(engineer_map='')
                ctrl.target = (9, 3)
                specs = (('Leo', (13, 3)), ('Derke', (2, 40)), ('Chronicle', (2, 18)),
                         ('Boaster', (2, 19)), ('Alfajer', (13, 23)))
                allies = [make_character(name, 'A', cell) for name, cell in specs]
                holder = allies[0]
                holder.has_spike = True
                for c in allies:
                    c.ability_name = 'NONE'
                    c.iq = c.effective_iq = 1
                game = UltimateTestGame(*grid.shape)
                game.grid = grid
                game.chars = allies
                adapter = IQAwareController(ctrl) if use_iq else ctrl
                adapter.set_game(game)
                game.attacker_controller = adapter
                # Keep the other members remote: planting must not depend on
                # an already-entering or flanking ally coming back to escort.
                for _ in range(12):
                    game.move_character(holder)
                    game.battle_tick += 1
                    if game.is_planted:
                        break
                self.assertTrue(game.is_planted)
                self.assertEqual(game.planted_pos, (9, 3))

    def test_changing_site_or_death_clears_committed_goal(self):
        for change in ('site', 'death'):
            grid, ctrl, holder, derke, allies = self.fixture()
            self.decide(grid, ctrl, derke, holder, allies)
            self.assertIsNotNone(ctrl.formation.rush_goal)
            if change == 'site':
                ctrl.target = (4, 2)
                derke.pos = [4, 22]
            else:
                derke.is_alive = False
            self.decide(grid, ctrl, holder, holder, allies)
            self.assertIsNone(ctrl.formation.rush_goal)

    def test_carrier_derke_and_spike_recovery_do_not_charge_past_spike(self):
        for recovery in (False, True):
            grid, ctrl, holder, derke, allies = self.fixture()
            if recovery:
                holder.has_spike = False
                destination = ctrl.target
            else:
                holder.name = 'Derke'
                derke.name = 'Entry'
                destination = None
            self.decide(grid, ctrl, holder, holder, allies, destination)
            self.assertIsNone(ctrl.formation.rush_goal)

    def test_committed_entry_reaches_every_registered_plant_with_team(self):
        grid = parse_grid(NEW_MAZE_STR)
        positions = FnaticV3AttackerController(engineer_map='').positions
        for target in positions.plant_cells:
            for use_iq in (False, True):
                with self.subTest(target=target, use_iq=use_iq):
                    ctrl = FnaticV3AttackerController(engineer_map='')
                    ctrl.target = target
                    game = UltimateTestGame(*grid.shape)
                    game.grid = grid
                    chars = [make_character(name, 'A', (23, 17 + index))
                             for index, name in enumerate(('Leo', 'Boaster', 'Derke', 'Chronicle', 'Alfajer'))]
                    chars[0].has_spike = True
                    game.chars = chars
                    controller = IQAwareController(ctrl) if use_iq else ctrl
                    controller.set_game(game)
                    game.attacker_controller = controller
                    committed = False
                    for _ in range(160):
                        for char in game._move_order():
                            game.move_character(char)
                        game.battle_tick += 1
                        self.assertEqual(len({tuple(c.pos) for c in chars}), 5)
                        committed |= ctrl.formation.rush_goal is not None
                        if game.is_planted:
                            break
                    self.assertTrue(committed)
                    self.assertTrue(game.is_planted, [(c.name, c.pos) for c in chars])
                    self.assertEqual(game.planted_pos, target)
                    self.assertEqual(ctrl.formation.mid_name, 'Alfajer')

    def test_reset_discards_committed_entry(self):
        grid, ctrl, holder, derke, allies = self.fixture()
        self.decide(grid, ctrl, derke, holder, allies)
        self.assertIsNotNone(ctrl.formation.rush_goal)
        ctrl.reset_round()
        self.assertIsNone(ctrl.formation.rush_goal)
        self.assertIsNone(ctrl.formation.rush_context)


if __name__ == '__main__':
    unittest.main()
