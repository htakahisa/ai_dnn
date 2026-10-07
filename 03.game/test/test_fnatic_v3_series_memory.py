"""Competition preserves the same tactical memory across maps and render paths."""

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_competition_manager as manager
from fnatic_v3.controller import FnaticV3AttackerController, FnaticV3DefenderController
from test_ultimate_system import UltimateTestGame, make_character


class SeriesBattle(UltimateTestGame):
    instances = []

    def __init__(self, maze, attacker_ai, defender_ai, **kwargs):
        super().__init__(14, 24)
        self.series_context = dict(kwargs['series_context'])
        self.attacker_team_name = kwargs['attacker_team_name']
        self.defender_team_name = kwargs['defender_team_name']
        self.attacker_roster = kwargs['attacker_roster']
        self.defender_roster = kwargs['defender_roster']
        self.attacker_wins, self.defender_wins = 13, 0
        side = 'A' if self.attacker_team_name == 'Fnatic2023' else 'D'
        self.ctrl = (FnaticV3AttackerController(engineer_map='') if side == 'A'
                     else FnaticV3DefenderController('', '', engineer_map=''))
        self.ctrl.set_game(self)
        self.chars = [make_character('Leo', side, (5, 3)),
                      make_character('Enemy', 'D' if side == 'A' else 'A', (4, 2))]
        self.root = SimpleNamespace(withdraw=lambda: None, update_idletasks=lambda: None,
                                    protocol=lambda *args: None, after=lambda *args: None)
        self.instances.append(self)

    def run(self):
        self.ctrl.opponent_history.observe(self.ctrl, self.chars[0], dict(
            grid=self.grid, chars=self.chars, is_planted=False, planted_pos=None))
        if self.ctrl.side == 'D':
            self.is_planted, self.planted_pos = True, (2, 21)
        self.ctrl.record_opponent_round_end()


class FnaticSeriesMemoryTests(unittest.TestCase):
    def setUp(self):
        SeriesBattle.instances = []

    def run_series(self, render=False):
        with patch.object(manager, 'VisualFPSBattle', SeriesBattle), \
                patch.object(manager, '_build_team_ai', return_value=None), \
                patch.object(manager, 'original_scores', return_value=(13, 0)), \
                patch.object(manager, 'seed_all'):
            return manager.run_series_core('Fnatic2023', 'Ghost Champions', 2, 'fixed', 1, 0,
                                           render, lambda event: None)

    def test_headless_maps_and_side_changes_keep_history_but_new_series_starts_empty(self):
        result = self.run_series()
        first, second = SeriesBattle.instances
        self.assertEqual(len(result.maps), 2)
        self.assertIs(first.series_context['fnatic_memory'], second.series_context['fnatic_memory'])
        self.assertIs(first.ctrl.opponent_history.data, second.ctrl.opponent_history.data)
        self.assertEqual(first.ctrl.opponent_history.data['attack_rounds'][0]['contacts']['Enemy']['region'], 'A')
        self.assertEqual(second.ctrl.opponent_history.data['defence_rounds'][0], dict(map=2, round=1, site='B'))
        self.run_series()
        third, fourth = SeriesBattle.instances[2:]
        self.assertIsNot(first.series_context['fnatic_memory'], third.series_context['fnatic_memory'])
        self.assertIs(third.series_context['fnatic_memory'], fourth.series_context['fnatic_memory'])
        self.assertEqual(len(third.ctrl.opponent_history.data['attack_rounds']), 1)

    def test_render_controller_headless_branch_passes_shared_series_memory(self):
        render = manager.CompetitionApp._RenderController(SimpleNamespace(live_render_enabled=False))
        self.run_series(render)
        first, second = SeriesBattle.instances
        self.assertIs(first.series_context['fnatic_memory'], second.series_context['fnatic_memory'])
        self.assertEqual(len(second.ctrl.opponent_history.data['attack_rounds']), 1)

    def test_render_queue_passes_same_memory_to_battle_constructor(self):
        app = SimpleNamespace(live_render_enabled=True, render_requests=queue.Queue())
        class ImmediateQueue:
            def put(self, request):
                app.render_requests.put(request)
                manager.CompetitionApp._process_render_requests(app)

        proxy_app = SimpleNamespace(live_render_enabled=True, render_requests=ImmediateQueue())
        render = manager.CompetitionApp._RenderController(proxy_app)
        self.run_series(render)
        first, second = SeriesBattle.instances
        self.assertIs(first.series_context['fnatic_memory'], second.series_context['fnatic_memory'])
        self.assertEqual(len(second.ctrl.opponent_history.data['attack_rounds']), 1)


if __name__ == '__main__':
    unittest.main()
