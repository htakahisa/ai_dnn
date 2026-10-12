"""Normal-game roster binding across side swaps; no optimizer updates."""
import contextlib
import io
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from touyama_v3 import tv3_game_controller as entry
from touyama_v3.tv3_character_stats_touyama import TOUYAMA_ROSTER_ORDER


class GameSideSwapTests(unittest.TestCase):
    def controller(self, side):
        cls = entry.TouyamaV3GameAttackerController if side == 'A' else entry.TouyamaV3GameDefenderController
        with patch.object(entry, 'Scenario', return_value=NS(grid=[])):
            controller = cls()
        controller.generic = Mock()
        controller._report = Mock()
        return controller

    def game(self, side):
        # Configured rosters have swapped; characters still reflect last round.
        rosters = dict(attacker_roster=['enemy']*5, defender_roster=['enemy']*5)
        rosters['attacker_roster' if side == 'A' else 'defender_roster'] = list(TOUYAMA_ROSTER_ORDER)
        return NS(**rosters, grid=[], chars=[NS(name='enemy',team=side)],
            current_attacker_team_ai=NS(name='FRC v1'), current_defender_team_ai=NS(name='FRC v1'))

    def test_both_roles_use_swapped_settings_before_characters_respawn(self):
        for side in ('A','D'):
            with self.subTest(side=side):
                controller = self.controller(side)
                game = self.game(side)
                if side == 'A':
                    with patch.object(entry,'load_sources',side_effect=FileNotFoundError('reached model loader')) as load:
                        controller.set_game(game)
                else:
                    with patch.object(entry,'load_policy',side_effect=FileNotFoundError('reached model loader')) as load:
                        controller.set_game(game)
                load.assert_called_once()
                self.assertIn('reached model loader',controller._report.call_args.args[1])

    def test_inactive_cached_role_does_not_rebind_to_its_own_team(self):
        for side in ('A','D'):
            controller = self.controller(side)
            game = self.game(side)
            team = game.current_attacker_team_ai if side=='A' else game.current_defender_team_ai
            setattr(team,'_attacker_controller' if side=='A' else '_defender_controller',object())
            controller.set_game(game)
            self.assertIsNone(controller.binding)
            controller.generic.set_game.assert_not_called()
            controller._report.assert_not_called()

    def test_roster_change_with_same_game_and_opponent_rechecks_binding(self):
        controller = self.controller('A')
        game = self.game('A')
        with patch.object(entry,'load_sources',side_effect=FileNotFoundError('missing')) as load:
            controller.set_game(game)
            game.attacker_roster = ['enemy']*5
            controller.set_game(game)
        load.assert_called_once()
        self.assertIn('味方編成が不一致',controller._report.call_args.args[1])

    def test_unconfigured_roster_uses_characters_and_duplicate_players_are_rejected(self):
        controller = self.controller('A')
        game = self.game('A')
        game.attacker_roster = None
        game.chars = [NS(name=n,base_name=n,team='A') for n in TOUYAMA_ROSTER_ORDER]
        with patch.object(entry,'load_sources',side_effect=FileNotFoundError('missing')) as load:
            controller.set_game(game)
            game.chars.append(game.chars[0])
            controller.set_game(game)
        load.assert_called_once()
        self.assertIn('味方編成が不一致',controller._report.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
