"""Search the real left-site blind corner through the IQ and combat engine."""

from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'omoko_v1', ROOT / 'test'):
    sys.path.insert(0, str(path))
from map_data import NEW_MAZE_STR
from ov1_inference_team_tactics import search_hidden_defuser, shot_clear
from ov1_attacker_controller import Ov1AttackerController
from ov1_learning_attacker_guard import Ov1LearningAttackerGuardController
from iq_controller_adapter import IQAwareController
from test_ultimate_system import UltimateTestGame, make_character


def real_grid():
    return np.array([[int(c) for c in row.strip()]
                     for row in NEW_MAZE_STR.strip().splitlines()])


class HiddenDefuseSearchTests(unittest.TestCase):
    def fixture(self, hidden_pos=(-1, -1)):
        actor = NS(name='ally', team='A', pos=[9, 3], facing='N', is_alive=True,
                   blind_remaining=0, facing_forced_this_tick=False)
        hidden = NS(name='defuser', team='D', pos=list(hidden_pos), facing='NW',
                    is_alive=True, position_known=False)
        state = dict(grid=real_grid(), chars=[actor, hidden], is_planted=True,
                     planted_pos=(9, 3), smoke_cells=(), current_round=1,
                     defender_defuse_info={'defuser': (2, 6)})
        return actor, hidden, state

    def test_reaches_peek_instead_of_rotating_at_spike(self):
        actor, hidden, state = self.fixture()
        position, payload = search_hidden_defuser(actor, state, {})
        self.assertEqual(position, [8, 3])
        self.assertEqual(actor._navigation_step_limit, 1)
        enemy = NS(name=hidden.name, pos=[8, 4], reveal_remaining=0)
        self.assertFalse(shot_clear(actor, enemy, state))
        self.assertTrue(shot_clear(actor, enemy, state, position))

    def test_hidden_position_and_occupancy_cannot_influence_search(self):
        results = []
        for hidden_pos in ((-1, -1), (8, 4), (8, 3), (3, 3)):
            actor, _, state = self.fixture(hidden_pos)
            memory = {}
            results.append((search_hidden_defuser(actor, state, memory), memory))
        self.assertTrue(all(result == results[0] for result in results))

    def test_known_defuser_uses_existing_firing_logic_and_no_tap_clears_memory(self):
        actor, hidden, state = self.fixture()
        memory = {}
        search_hidden_defuser(actor, state, memory)
        state['defender_defuse_info'] = {}
        self.assertIsNone(search_hidden_defuser(actor, state, memory))
        self.assertEqual(memory, {})
        state['defender_defuse_info'] = {'defuser': (1, 6)}
        hidden.position_known, hidden.pos = True, [8, 4]
        self.assertIsNone(search_hidden_defuser(actor, state, memory))

    def test_cancelled_and_restarted_tap_resets_checked_cells(self):
        actor, _, state = self.fixture()
        memory = {}
        search_hidden_defuser(actor, state, memory)
        memory['remaining'] = {(8, 4)}
        state['defender_defuse_info'] = {'defuser': (1, 6)}
        search_hidden_defuser(actor, state, memory)
        self.assertNotEqual(memory['remaining'], {(8, 4)})

    def test_does_not_walk_into_occupied_peek_or_override_forced_facing(self):
        actor, _, state = self.fixture()
        actor.facing_forced_this_tick, actor.facing = True, 'S'
        state['chars'].append(NS(name='blocker', team='A', pos=[8, 3], is_alive=True))
        result = search_hidden_defuser(actor, state, {})
        self.assertNotEqual(result[0], [8, 3])
        self.assertEqual(actor.facing, 'S')

    def test_engine_three_survivors_interrupt_left_corner_before_six_ticks(self):
        for hp in (20, 70, 100):
            with self.subTest(hp=hp):
                self.run_engine_case(hp)

    def test_previous_behavior_reproduces_the_six_tick_defuse(self):
        with patch('ov1_attacker_controller.search_hidden_defuser', return_value=None), \
                self.assertRaisesRegex(AssertionError, 'True is not false'):
            self.run_engine_case(20)

    def test_wrapper_search_keeps_public_ash_entry_avoidance(self):
        from public_effects import DisplayEffect
        actor, _, state = self.fixture()
        actor.has_spike = False
        state['public_effects'] = (
            DisplayEffect(1, 'DESTRUCTION', 'active', (8, 3), cells=((8, 3),)),)
        ctrl = Ov1AttackerController.__new__(Ov1AttackerController)
        position, _ = ctrl.decide_move(actor, state)
        self.assertNotEqual(position, [8, 3])

    def run_engine_case(self, hp, random_roll=None):
        grid = real_grid()
        game = UltimateTestGame(height=grid.shape[0], width=grid.shape[1])
        game.grid = grid
        allies = [make_character(name, 'A', pos) for name, pos in
                  (('\u306d\u3053\u3055\u3093', (10, 2)),
                   ('\u3044\u306c\u3055\u3093', (13, 3)),
                   ('\u3072\u3064\u3058\u3055\u3093', (13, 7)))]
        for actor, facing in zip(allies, ('NE', 'N', 'N')):
            actor.facing = facing
        enemy = make_character('Lohen', 'D', (8, 4))
        enemy.hp, enemy.facing, enemy.los_revealed = hp, 'NW', False
        game.chars = allies + [enemy]
        game.is_planted, game.is_defused, game.planted_pos = True, False, (9, 3)
        game.active_defuser_name = enemy.name
        ctrl = Ov1AttackerController.__new__(Ov1AttackerController)
        for phase in ('carry', 'escort', 'retrieve', 'guard'):
            setattr(ctrl, phase + '_controller', Mock())
        ctrl.guard_controller = Ov1LearningAttackerGuardController.__new__(Ov1LearningAttackerGuardController)
        wrapper = IQAwareController(ctrl, viewer_team='A')
        wrapper.set_game(game)
        game.attacker_controller = wrapper
        visited = []
        roll = random_roll or (lambda: 0.0)
        with patch.object(wrapper.perception_engine, '_omit_enemy', return_value=False), \
                patch('battle_logic.random.random', side_effect=roll):
            for tick in range(1, 7):
                game.battle_tick = tick
                enemy.defuse_timer = tick
                for actor in allies:
                    game.move_character(actor)
                visited.append(tuple(allies[0].pos))
                game._resolve_all_shots()
                game._resolve_defuse_completion()
                self.assertFalse(game.is_defused, (tick, visited))
                if not enemy.is_alive:
                    break
                game._update_los_reveal()
        self.assertIn((8, 3), visited)
        self.assertFalse(enemy.is_alive, visited)
        self.assertLessEqual(tick, 6)
        return tick


if __name__ == '__main__':
    unittest.main()
