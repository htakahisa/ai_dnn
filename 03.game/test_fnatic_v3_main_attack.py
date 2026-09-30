"""Team-visible, three-attacker reads redirect mapped pre-plant defense."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fnatic_v3.attacker_ability import FnaticEngineerRoute
from fnatic_v3.positions import distances, region
from iq_perception import PerceivedCharacter
import test_fnatic_v3_defender_positions as fixtures
from test_fnatic_v3_defender_positions import placement_map
from test_ultimate_system import make_character


class FnaticMainAttackTests(unittest.TestCase):
    def fixture(self):
        candidates = {3: tuple((2, c) for c in (1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21))}
        game, ctrl = fixtures.FnaticDefenderPositionTests().fixture(candidates=candidates)
        return game, ctrl, list(game.chars)

    @staticmethod
    def state(game, chars=None, **fields):
        state = fixtures.FnaticDefenderPositionTests.state(game)
        if chars is not None:
            state['chars'] = chars
        state.update(fields)
        return state

    @staticmethod
    def enemies(site, count=3):
        col = {'A': 2, 'MID': 10, 'B': 18}[site]
        return [make_character('Enemy' + str(i), 'A', (6, col + i)) for i in range(count)]

    def assert_focused(self, game, ctrl, label, count=5):
        targets = ctrl.defender_positions.targets
        self.assertEqual(len(targets), count)
        self.assertEqual(len(set(targets.values())), count)
        self.assertTrue(all(region(cell, game.grid) == label for cell in targets.values()))
        self.assertTrue(all(cell in ctrl.defender_positions.candidates[3] for cell in targets.values()))

    def test_three_visible_enemies_redirect_all_living_players_on_both_sites(self):
        for site in ('A', 'B'):
            with self.subTest(site=site):
                game, ctrl, allies = self.fixture()
                ctrl.decide_move(allies[0], self.state(game))
                game.chars.extend(self.enemies(site))
                game.battle_tick += 1
                ctrl.decide_move(allies[-1], self.state(game))
                self.assertEqual(ctrl.defender_positions.main_region, site)
                self.assert_focused(game, ctrl, site)
                targets = dict(ctrl.defender_positions.targets)
                game.battle_tick += 1
                # Remote players can act on the team's report after losing sight.
                for ally in allies:
                    before = tuple(ally.pos)
                    result = ctrl.decide_move(ally, self.state(game, allies))
                    lengths = distances(targets[ally.name], game.grid)
                    self.assertLess(lengths[tuple(result[0])], lengths[before])
                self.assertEqual(ctrl.defender_positions.targets, targets)

    def test_two_enemies_mid_and_dead_enemies_do_not_trigger(self):
        for site, count, dead in (('A', 2, False), ('B', 2, False), ('MID', 3, False), ('A', 3, True)):
            with self.subTest(site=site, count=count, dead=dead):
                game, ctrl, allies = self.fixture()
                enemies = self.enemies(site, count)
                if dead:
                    enemies[-1].is_alive = False
                game.chars.extend(enemies)
                ctrl.decide_move(allies[0], self.state(game))
                self.assertIsNone(ctrl.defender_positions.main_region)

    def test_hidden_enemies_behind_walls_do_not_trigger(self):
        game, ctrl, allies = self.fixture()
        game.grid[:, 8] = 1
        ctrl.defender_positions = type(ctrl.defender_positions)(placement_map(
            game.grid, {3: ((2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 20))}))
        game.chars.extend(self.enemies('B'))
        ctrl.decide_move(allies[0], self.state(game))
        self.assertIsNone(ctrl.defender_positions.main_region)

    def test_shared_reports_use_perceived_positions_and_count_each_enemy_once(self):
        game, ctrl, allies = self.fixture()
        real_enemies = self.enemies('A')
        game.chars.extend(real_enemies)
        engine = SimpleNamespace(build_game_view=Mock())
        game.defender_controller = SimpleNamespace(perception_engine=engine)
        reports = {
            allies[0].name: [PerceivedCharacter(real_enemies[0], pos=[6, 18])],
            allies[1].name: [PerceivedCharacter(real_enemies[1], pos=[6, 19])],
            allies[2].name: [PerceivedCharacter(real_enemies[2], pos=[6, 20])],
        }
        engine.build_game_view.side_effect = lambda viewer, game: SimpleNamespace(chars=reports.get(viewer.name, []))
        ctrl.decide_move(allies[-1], self.state(game, allies))
        self.assertEqual(ctrl.defender_positions.main_region, 'B')
        self.assert_focused(game, ctrl, 'B')
        self.assertEqual(engine.build_game_view.call_count, 5)

        game, ctrl, allies = self.fixture()
        enemies = self.enemies('B', 2)
        game.chars.extend(enemies)
        engine = SimpleNamespace(build_game_view=lambda **kwargs: SimpleNamespace(chars=enemies))
        game.defender_controller = SimpleNamespace(perception_engine=engine)
        ctrl.decide_move(allies[0], self.state(game, allies))
        self.assertIsNone(ctrl.defender_positions.main_region)

    def test_previous_sightings_do_not_accumulate_into_three(self):
        game, ctrl, allies = self.fixture()
        enemies = self.enemies('A')
        ctrl.decide_move(allies[0], self.state(game, allies + enemies[:2]))
        game.battle_tick += 1
        ctrl.decide_move(allies[0], self.state(game, allies + enemies[2:]))
        self.assertIsNone(ctrl.defender_positions.main_region)

    def test_read_latches_then_changes_on_a_fresh_report_and_resets_each_round(self):
        game, ctrl, allies = self.fixture()
        ctrl.decide_move(allies[0], self.state(game, allies + self.enemies('A')))
        self.assert_focused(game, ctrl, 'A')
        game.battle_tick += 1
        ctrl.decide_move(allies[0], self.state(game, allies))
        self.assert_focused(game, ctrl, 'A')
        game.battle_tick += 1
        ctrl.decide_move(allies[0], self.state(game, allies + self.enemies('B')))
        self.assert_focused(game, ctrl, 'B')
        ctrl.reset_round()
        self.assertIsNone(ctrl.defender_positions.main_region)
        self.assertIsNone(ctrl.defender_positions.main_report_tick)

    def test_main_read_overrides_history_lamp_distribution_and_unfinished_lamp_route(self):
        game, ctrl, allies = self.fixture()
        ctrl.opponent_history.bind(ctrl)
        ctrl.opponent_history.data['defence_rounds'] = [{'site': 'A'}]
        game.ramp_traps = [dict(pos=(3, 20), owner='Alfajer', team='D')]
        ctrl.engineer = FnaticEngineerRoute(placement_map(game.grid, {3: ((4, 2), (4, 3))}))
        ctrl.decide_move(allies[0], self.state(game, allies + self.enemies('B')))
        self.assert_focused(game, ctrl, 'B')
        self.assertIsNone(ctrl.defender_positions.region_limits)
        alf = allies[0]
        game.battle_tick += 1
        result = ctrl.decide_move(alf, self.state(game, allies))
        self.assertEqual(alf.ramp_charges, 2)
        self.assertIsNone(ctrl.engineer.target)
        self.assertNotEqual(result[0], alf.pos)

    def test_dropped_spike_still_overrides_the_main_read(self):
        game, ctrl, allies = self.fixture()
        game.spike_pos = (4, 2)
        ctrl.decide_move(allies[0], self.state(game, allies + self.enemies('B')))
        self.assertEqual(ctrl.defender_positions.main_region, 'B')
        self.assert_focused(game, ctrl, 'A')
        game.spike_pos = None
        game.battle_tick += 1
        ctrl.decide_move(allies[0], self.state(game, allies))
        self.assert_focused(game, ctrl, 'B')

    def test_setup_and_postplant_do_not_trigger_preplant_read(self):
        for setup, planted in ((True, False), (False, True)):
            with self.subTest(setup=setup, planted=planted):
                game, ctrl, allies = self.fixture()
                game.is_planted = planted
                game.planted_pos = (5, 2) if planted else None
                ctrl.decide_move(allies[0], self.state(game, allies + self.enemies('B'), defender_setup_active=setup))
                self.assertIsNone(ctrl.defender_positions.main_region)

    def test_dead_allies_release_assignments_after_the_read(self):
        game, ctrl, allies = self.fixture()
        allies[-1].is_alive = False
        ctrl.decide_move(allies[0], self.state(game, allies + self.enemies('B')))
        self.assert_focused(game, ctrl, 'B', count=4)
        self.assertNotIn(allies[-1].name, ctrl.defender_positions.targets)


if __name__ == '__main__':
    unittest.main()
