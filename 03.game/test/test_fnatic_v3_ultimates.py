"""Fnatic ultimate rules exercised through real game effects and perception."""

import unittest

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController, FnaticV3DefenderController
from fnatic_v3.positions import TacticalPositions, distances
from fnatic_v3.ultimate_tactics import region
from iq_controller_adapter import IQAwareController
from test_fnatic_v3_defender_positions import placement_map
from test_ultimate_system import UltimateTestGame, make_character


class FnaticUltimateTests(unittest.TestCase):
    def fixture(self, caster='Chronicle', side='A', use_iq=False):
        grid = np.zeros((12, 36), dtype=np.int32)
        grid[3:7, 3:7] = 2
        grid[3:7, 29:33] = 2
        grid[10, 14:19] = 3
        grid[1, 14:19] = 4
        marked = np.where(grid == 1, 1, 0)
        marked[5, 4] = 5
        marked[5, 30] = 6
        text = '\n'.join(''.join(map(str, row)) for row in marked)
        ctrl = (FnaticV3AttackerController(TacticalPositions(text, text), engineer_map='')
                if side == 'A' else FnaticV3DefenderController(position_map='', retake_map='', engineer_map=''))
        ctrl.target = (5, 30)
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        specs = [('Leo', (5, 24)), ('Derke', (4, 25)), ('Chronicle', (5, 25)),
                 ('Boaster', (6, 24)), ('Alfajer', (6, 25))]
        game.chars = [make_character(name, side, cell) for name, cell in specs]
        for char in game.chars:
            char.iq = char.effective_iq = 200
        if side == 'A':
            game.chars[0].has_spike = True
        char = next(c for c in game.chars if c.name == caster)
        char.ultimate_points = char.ultimate_cost
        char.facing = 'W'
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        if side == 'A':
            game.attacker_controller = adapter
        else:
            game.defender_controller = adapter
        return game, ctrl, char

    @staticmethod
    def state(game):
        return dict(grid=game.grid, chars=game.chars, round_timer=100,
                    is_planted=game.is_planted, planted_pos=game.planted_pos,
                    defender_defuse_info={c.name: (c.defuse_timer, 6) for c in game.chars
                                         if c.team == 'D' and c.is_alive})

    def test_entry_tunnel_and_neon_cover_site_and_spend_points(self):
        for caster in ('Chronicle', 'Alfajer'):
            for use_iq in (False, True):
                with self.subTest(caster=caster, use_iq=use_iq):
                    game, _, char = self.fixture(caster, use_iq=use_iq)
                    enemy = make_character('Enemy', 'D', (5, 31))
                    game.chars.append(enemy)
                    game.move_character(char)
                    self.assertEqual(char.ultimate_points, 0)
                    if caster == 'Chronicle':
                        self.assertEqual(char.facing, 'E')
                        self.assertIn((5, 31), game.tunnel_bursts[0]['cells'])
                        for _ in range(6):
                            game._advance_tunnel_bursts()
                        self.assertGreater(enemy.blind_remaining, 0)
                    else:
                        target = game.neon_bursts[0]['pos']
                        self.assertEqual(game.grid[target], 2)
                        for _ in range(11):
                            game._advance_engineer_effects()
                        self.assertLess(enemy.hp, enemy.max_hp)

    def test_monitor_casts_at_twenty_ticks_on_both_sides_with_and_without_iq(self):
        for side in ('A', 'D'):
            for use_iq in (False, True):
                with self.subTest(side=side, use_iq=use_iq):
                    game, _, char = self.fixture('Leo', side, use_iq=use_iq)
                    # No entry, target contact or visible enemies is required.
                    char.has_spike = False
                    game.chars = [char]
                    char.pos = [10, 16]
                    game.battle_tick = 19
                    game.move_character(char)
                    self.assertEqual(char.ultimate_points, char.ultimate_cost)
                    self.assertEqual(game.monitor_drones, [])
                    game.battle_tick = 20
                    game.move_character(char)
                    self.assertEqual(char.ultimate_points, 0)
                    self.assertEqual(len(game.monitor_drones), 2)
                    game.move_character(char)
                    self.assertEqual(len(game.monitor_drones), 2)

    def test_monitor_casts_when_points_become_ready_after_twenty_ticks(self):
        game, _, char = self.fixture('Leo')
        game.battle_tick = 35
        char.ultimate_points -= 1
        game.move_character(char)
        self.assertEqual(game.monitor_drones, [])
        char.ultimate_points = char.ultimate_cost
        game.move_character(char)
        self.assertEqual(char.ultimate_points, 0)
        self.assertEqual(len(game.monitor_drones), 2)

    def test_monitor_new_round_uses_elapsed_ticks_instead_of_remaining_timer(self):
        game, ctrl, char = self.fixture('Leo')
        game.battle_tick = 40
        game.move_character(char)
        ctrl.reset_round()
        game.battle_tick = 0
        game.round_timer = 30
        char.ultimate_points = char.ultimate_cost
        state = self.state(game)
        # Neither a shortened round timer nor stale perceived tick information
        # should make the monitor available before the next round's 20 ticks.
        state['battle_tick'] = 40
        self.assertIsNone(ctrl.ultimates.result(ctrl, char, state))

    def test_monitor_can_cast_while_responding_to_defuse_notification(self):
        game, ctrl, char = self.fixture('Leo')
        game.battle_tick = 20
        game.is_planted = True
        game.planted_pos = (5, 30)
        char.has_spike = False
        game.chars = [char]
        enemy = make_character('Enemy', 'D', (5, 30))
        enemy.defuse_timer = 1
        game.chars.append(enemy)
        game.move_character(char)
        self.assertEqual(ctrl.defuse_responder, char.name)
        self.assertEqual(char.ultimate_points, 0)
        self.assertEqual(len(game.monitor_drones), 2)

    def test_entry_raid_moves_toward_site_with_requested_facing(self):
        game, _, char = self.fixture('Derke')
        before = min(distances(tuple(char.pos), game.grid)[p] for p in
                     zip(*np.where(game.grid[:, :] == 2)) if p[1] > 20)
        game.move_character(char)
        after = min(distances(tuple(char.pos), game.grid)[p] for p in
                    zip(*np.where(game.grid[:, :] == 2)) if p[1] > 20)
        self.assertLess(after, before)
        self.assertNotEqual(char.facing, 'W')
        self.assertEqual(char.ultimate_points, 0)
        self.assertEqual(len(game.ultimate_trails), 1)

    def test_entry_escape_lands_inside_site_after_windup(self):
        game, ctrl, char = self.fixture('Boaster')
        game.move_character(char)
        portal = game.escape_portals[0]
        target = portal['pos']
        self.assertEqual(game.grid[target], 2)
        self.assertNotEqual(target, ctrl.target)  # Leave the carrier's plant cell free.
        self.assertEqual(char.ultimate_points, 0)
        for _ in range(10):
            game._advance_escape_portals()
            self.assertNotEqual(tuple(char.pos), target)
        game._advance_escape_portals()
        self.assertEqual(tuple(char.pos), target)

    def test_defender_retakes_use_tunnel_and_neon_inside_site(self):
        for caster in ('Chronicle', 'Alfajer'):
            game, _, char = self.fixture(caster, 'D')
            for ally in game.chars:
                ally.ability_name = 'NONE'  # Test ultimate entry after utility is exhausted.
            game.is_planted = True
            game.planted_pos = (5, 30)
            game.move_character(char)
            self.assertEqual(char.ultimate_points, 0)
            if caster == 'Chronicle':
                self.assertIn((5, 30), game.tunnel_bursts[0]['cells'])
            else:
                self.assertEqual(game.grid[game.neon_bursts[0]['pos']], 2)

    def test_postplant_ultimates_aim_at_seen_enemy_and_wait_without_one(self):
        for caster in ('Chronicle', 'Alfajer'):
            game, ctrl, char = self.fixture(caster)
            game.is_planted = True
            game.planted_pos = (5, 30)
            self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))
            enemy = make_character('Enemy', 'D', (8, 25))
            game.chars.append(enemy)
            game.move_character(char)
            self.assertEqual(char.ultimate_points, 0)
            if caster == 'Chronicle':
                self.assertIn(tuple(enemy.pos), game.tunnel_bursts[0]['cells'])
            else:
                self.assertEqual(game.neon_bursts[0]['pos'], tuple(enemy.pos))

    def test_visible_defuser_can_be_stopped_by_responder_ultimate(self):
        game, ctrl, char = self.fixture('Chronicle')
        game.chars = [char]
        char.pos = [5, 28]
        game.is_planted = True
        game.planted_pos = (5, 30)
        enemy = make_character('Enemy', 'D', (5, 30))
        enemy.defuse_timer = 1
        game.chars.append(enemy)
        game.move_character(char)
        self.assertEqual(ctrl.defuse_responder, char.name)
        self.assertEqual(char.ultimate_points, 0)
        self.assertIn(tuple(enemy.pos), game.tunnel_bursts[0]['cells'])

    def test_postkill_raid_escapes_two_seen_enemies_into_cover(self):
        game, ctrl, char = self.fixture('Derke', 'D')
        char.pos = [4, 7]
        char.ultimate_points = char.ultimate_cost - 1
        enemies = [make_character('Enemy1', 'A', (4, 11)),
                   make_character('Enemy2', 'A', (5, 11))]
        victim = make_character('Victim', 'A', (4, 8))
        game.chars = [char, victim, *enemies]
        game.grid[6, 8:16] = 1
        game._kill_character(char, victim)
        self.assertTrue(all(game.check_line_of_sight(char, c) for c in enemies))
        game.move_character(char)
        self.assertEqual(char.ultimate_points, 0)
        self.assertTrue(all(not game.check_line_of_sight(char, c) for c in enemies))
        # The same old kill cannot trigger a later retreat.
        char.ultimate_points = char.ultimate_cost
        game.battle_tick += 3
        self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))

    def test_postkill_escape_requires_two_enemies_and_a_cover_direction(self):
        for enemy_count, kill_count in ((1, 1), (2, 0), (2, 1)):
            game, ctrl, char = self.fixture('Derke', 'D')
            char.pos = [5, 20]
            char.round_kills = kill_count
            game.chars = [char] + [make_character('Enemy' + str(i), 'A', (5 + i, 25))
                                  for i in range(enemy_count)]
            # An open map offers no direction that cuts all incoming lines.
            self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))
            self.assertEqual(char.ultimate_points, char.ultimate_cost)

    def test_ally_contact_escape_flanks_to_enemy_spawn_on_both_sides(self):
        for side in ('A', 'D'):
            game, ctrl, char = self.fixture('Boaster', side)
            char.pos = [5, 2]
            ally = make_character('Derke', side, (7, 20))
            enemy = make_character('Enemy', 'D' if side == 'A' else 'A', (7, 24))
            game.chars = [char, ally, enemy]
            game.grid[:, 10] = 1
            if side == 'A':
                ctrl.positions.plant_grid[:, 10] = 1
                ctrl.positions.guard_grid[:, 10] = 1
            self.assertFalse(game.check_line_of_sight(char, enemy))
            self.assertTrue(game.check_line_of_sight(ally, enemy))
            game.is_planted = True
            game.planted_pos = (5, 30)
            action = ctrl.ultimates.result(ctrl, char, self.state(game))
            if side == 'D':
                self.assertIsNone(action)
            else:
                self.assertEqual(action[1]['ultimate'], 'ESCAPE')
                self.assertEqual(game.grid[tuple(action[1]['target'])], 4)
            game.is_planted = False
            game.planted_pos = None
            game.move_character(char)
            target = game.escape_portals[0]['pos']
            self.assertEqual(game.grid[target], 4 if side == 'A' else 3)
            self.assertEqual(char.ultimate_points, 0)

    def reinforcement(self, col=26, enemies=None, use_iq=False):
        game, _, char = self.fixture('Boaster', 'D')
        game.grid[(game.grid == 3) | (game.grid == 4)] = 0
        char.pos = [5, col]
        ally = make_character('Derke', 'D', (5, 7))
        ally.iq = ally.effective_iq = 200
        enemy_positions = enemies if enemies is not None else ((4, 4), (5, 4), (6, 4))
        enemy_chars = [make_character('Enemy' + str(i), 'A', cell) for i, cell in enumerate(enemy_positions)]
        game.chars = [char, ally, *enemy_chars]
        candidates = {3: ((4, 7),), 5: ((7, 4),), 9: ((10, 11),)}
        ctrl = FnaticV3DefenderController(placement_map(game.grid, candidates), retake_map='', engineer_map='')
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        game.defender_controller = adapter
        return game, ctrl, char

    def test_defender_three_seen_enemies_and_twenty_steps_rotate_to_mapped_slot(self):
        game, ctrl, char = self.reinforcement()
        action = ctrl.decide_move(char, self.state(game))
        self.assertEqual(action[1]['ultimate'], 'ESCAPE')
        self.assertIn(action[1]['target'], {(4, 7), (7, 4)})
        self.assertTrue(game.execute_ai_ultimate(char, action[1]))
        target = action[1]['target']
        game.battle_tick += 1
        ctrl.decide_move(char, self.state(game))
        self.assertEqual(ctrl.defender_positions.targets[char.name], target)
        self.assertEqual(len(set(ctrl.defender_positions.targets.values())), len(ctrl.defender_positions.targets))

    def test_reinforcement_conditions_use_distinct_enemies_in_same_region(self):
        for col, enemies in ((25, ((4, 4), (5, 4), (6, 4))),
                             (26, ((4, 4),)),
                             (26, ((4, 4), (5, 4), (5, 30)))):
            game, ctrl, char = self.reinforcement(col, enemies)
            # Several allies seeing the same enemy still count as one.
            game.chars.insert(1, make_character('Chronicle', 'D', (7, 7)))
            game.chars.insert(1, make_character('Leo', 'D', (8, 7)))
            self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))

    def test_reinforcement_distance_includes_wall_detour(self):
        game, ctrl, char = self.reinforcement(col=15)
        # Direct separation is only nine cells, but walking around this wall
        # takes twenty steps to the nearest cell of the A plant area.
        game.grid[:11, 12] = 1
        ctrl.defender_positions.grid[:11, 12] = 1
        plants = [p for p in zip(*np.where(game.grid == 2)) if p[1] < 12]
        self.assertEqual(min(distances(tuple(char.pos), game.grid)[p] for p in plants), 20)
        action = ctrl.ultimates.result(ctrl, char, self.state(game))
        self.assertIn(action[1]['target'], {(4, 7), (7, 4)})

    def test_reinforcement_also_uses_b_region_candidates(self):
        game, _, char = self.reinforcement()
        game.grid = np.fliplr(game.grid).copy()
        for c in game.chars:
            c.pos[1] = game.width - 1 - c.pos[1]
        ctrl = FnaticV3DefenderController(placement_map(game.grid, {
            3: ((4, 28),), 5: ((7, 31),), 9: ((10, 24),)}), retake_map='', engineer_map='')
        ctrl.set_game(game)
        action = ctrl.ultimates.result(ctrl, char, self.state(game))
        self.assertIn(action[1]['target'], {(4, 28), (7, 31)})
        self.assertTrue(game.execute_ai_ultimate(char, action[1]))

    def test_mid_main_uses_seen_positions_when_there_is_no_plant_area(self):
        game, _, char = self.reinforcement(col=1, enemies=((3, 18), (4, 18), (5, 18)))
        char.pos = [7, 1]
        game.chars[1].pos = [5, 20]
        ctrl = FnaticV3DefenderController(placement_map(game.grid, {
            3: ((4, 16),), 9: ((11, 12),)}), retake_map='', engineer_map='')
        ctrl.set_game(game)
        # The closest observed enemy is 19 steps away: hold the ultimate.
        self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))
        char.pos = [7, 0]
        action = ctrl.ultimates.result(ctrl, char, self.state(game))
        self.assertEqual(action[1]['target'], (4, 16))

    def test_walls_and_iq_do_not_hide_a_teammates_report_from_remote_caster(self):
        game, ctrl, char = self.reinforcement(use_iq=True)
        game.grid[:, 20] = 1
        # Rebuild only the test placement map to match the added barrier.
        ctrl.defender_positions.grid[:, 20] = 1
        char.iq = char.effective_iq = 10
        game.move_character(char)
        self.assertEqual(char.ultimate_points, 0)
        self.assertIn(game.escape_portals[0]['pos'], {(4, 7), (7, 4)})

    def test_unseen_enemies_cannot_trigger_defensive_rotation(self):
        game, ctrl, char = self.reinforcement()
        # Put every observer on the opposite side of an opaque barrier.
        game.chars[1].pos = [7, 25]
        game.grid[:, 20] = 1
        ctrl.defender_positions.grid[:, 20] = 1
        self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))

    def test_rejected_escape_does_not_change_defensive_assignment(self):
        game, ctrl, char = self.reinforcement()
        action = ctrl.decide_move(char, self.state(game))
        original = dict(ctrl.defender_positions.targets)
        # Rejected by the engine after another player occupies the landing.
        blocker = make_character('Blocker', 'D', action[1]['target'])
        game.chars.append(blocker)
        self.assertFalse(game.execute_ai_ultimate(char, action[1]))
        ctrl.ultimates.sync(ctrl, char)
        self.assertEqual(ctrl.defender_positions.targets, original)

    def test_active_portal_and_electric_status_prevent_duplicate_or_invalid_casts(self):
        for caster in ('Derke', 'Boaster'):
            game, ctrl, char = self.fixture(caster)
            char.electric_remaining = 2
            self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))
        game, ctrl, char = self.fixture('Boaster')
        game.move_character(char)
        char.ultimate_points = char.ultimate_cost
        self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))

    def test_setup_plant_and_defuse_are_not_interrupted(self):
        for phase in ('setup', 'plant', 'defuse'):
            game, ctrl, char = self.fixture('Chronicle', 'D' if phase == 'defuse' else 'A')
            if phase == 'plant':
                game.chars[0].has_spike = False
                char.has_spike = True
                char.pos = list(ctrl.target)
            if phase == 'defuse':
                game.is_planted = True
                game.planted_pos = tuple(char.pos)
            state = self.state(game)
            if phase == 'setup':
                state['defender_setup_active'] = True
            action = ctrl.decide_move(char, state)
            self.assertFalse(isinstance(action[1], dict) and 'ultimate' in action[1])
            self.assertEqual(char.ultimate_points, char.ultimate_cost)

    def test_not_ready_and_no_entry_keep_points(self):
        game, ctrl, char = self.fixture('Chronicle')
        char.ultimate_points -= 1
        self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))
        char.ultimate_points = char.ultimate_cost
        game.chars[0].pos = [10, 16]
        self.assertIsNone(ctrl.ultimates.result(ctrl, char, self.state(game)))

    def test_round_reset_clears_kill_and_relocation_state(self):
        game, ctrl, char = self.fixture('Derke')
        char.round_kills = 1
        ctrl.ultimates.result(ctrl, char, self.state(game))
        ctrl.ultimates.relocations[char.name] = ((2, 2), 2)
        ctrl.reset_round()
        self.assertEqual(ctrl.ultimates.kill_counts, {})
        self.assertEqual(ctrl.ultimates.relocations, {})

    def test_region_classification_is_a_mid_b(self):
        game, _, _ = self.fixture()
        self.assertEqual([region((5, c), game.grid) for c in (4, 18, 30)], ['A', 'MID', 'B'])


if __name__ == '__main__':
    unittest.main()
