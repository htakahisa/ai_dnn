"""Live controller boundaries must never pass a hidden enemy coordinate."""

import unittest
from types import SimpleNamespace

import numpy as np

from iq_controller_adapter import IQAwareController
from iq_perception import IQPerceptionEngine, UNKNOWN_ENEMY_POS, build_team_position_view
from team_ai import PrivateInfoController
from test_ultimate_system import UltimateTestGame, make_character


class RecordingController:
    def __init__(self):
        self.games = []
        self.seen = None

    def set_game(self, game):
        self.games.append(game)

    def decide_move(self, char, state):
        self.seen = (char, state)
        return list(char.pos)


def fixture():
    ally = SimpleNamespace(name="ally", team="A", pos=[1, 1], hp=100,
                           is_alive=True, iq=200, effective_iq=200,
                           has_spike=False, defuse_timer=0, reveal_remaining=0,
                           los_revealed=False)
    enemy = SimpleNamespace(name="enemy", team="D", pos=[7, 8], hp=100,
                            is_alive=True, iq=200, effective_iq=200,
                            has_spike=True, defuse_timer=0, reveal_remaining=0,
                            los_revealed=False)
    game = SimpleNamespace(chars=[ally, enemy], grid=np.zeros((10, 10), dtype=int),
                           current_round=1, battle_tick=1, spike_pos=None,
                           planted_pos=None, target_plant_pos=None, is_planted=False,
                           round_timer=30, detonate_timer=0, visible=False,
                           last_engagements=((ally, enemy),))
    game.is_visible_to_team = lambda target, team: game.visible or target.reveal_remaining > 0
    return game, ally, enemy


class EnemyPositionVisibilityTests(unittest.TestCase):
    def assert_hidden(self, game, state):
        enemy = state["chars"][1]
        self.assertEqual(tuple(enemy.pos), UNKNOWN_ENEMY_POS)
        self.assertFalse(enemy.position_known)
        self.assertEqual(tuple(enemy.real_character.pos), UNKNOWN_ENEMY_POS)
        self.assertEqual(tuple(game.chars[1].pos), (7, 8))

    def test_iq_controller_hides_enemy_at_all_input_paths(self):
        game, ally, enemy = fixture()
        inner = RecordingController()
        controller = IQAwareController(inner, IQPerceptionEngine())
        controller.set_game(game)
        self.assertEqual(inner.games, [])
        controller.decide_move(ally, {"chars": game.chars})
        self.assert_hidden(game, inner.seen[1])
        view = inner.games[-1]
        self.assertIs(view.real_game, view)
        self.assertEqual(tuple(view.last_engagements[0][1].pos), UNKNOWN_ENEMY_POS)
        enemy.reveal_remaining = 1
        game.battle_tick += 1
        controller.decide_move(ally, {"chars": game.chars})
        self.assertEqual(tuple(inner.seen[1]["chars"][1].pos), (7, 8))
        # Visibility can change while other characters move in the same tick.
        enemy.reveal_remaining = 0
        controller.decide_move(ally, {"chars": game.chars})
        self.assert_hidden(game, inner.seen[1])

    def test_non_iq_controller_hides_enemy_and_keeps_visible_enemy(self):
        game, ally, enemy = fixture()
        inner = RecordingController()
        controller = PrivateInfoController(inner)
        controller.set_game(game)
        self.assertEqual(inner.games, [])
        controller.decide_move(ally, {"chars": game.chars})
        self.assert_hidden(game, inner.seen[1])
        self.assertIs(inner.games[-1].real_game, inner.games[-1])
        game.visible = True  # Raw LOS alone must not publish a position.
        controller.decide_move(ally, {"chars": game.chars})
        self.assert_hidden(game, inner.seen[1])
        enemy.los_revealed = True
        controller.decide_move(ally, {"chars": game.chars})
        self.assertEqual(tuple(inner.seen[1]["chars"][1].pos), (7, 8))
        self.assertTrue(inner.seen[1]["chars"][1].position_known)

    def test_setup_view_uses_same_position_rule(self):
        game, _, enemy = fixture()
        view = build_team_position_view(game, "A")
        self.assertEqual(tuple(view.chars[1].pos), UNKNOWN_ENEMY_POS)
        enemy.reveal_remaining = 1
        self.assertEqual(tuple(build_team_position_view(game, "A").chars[1].pos), (7, 8))

    def test_pre_round_binding_only_passes_filtered_game(self):
        game, _, _ = fixture()
        game.replay_frames = [{"chars": [{"pos": [7, 8]}]}]
        for controller in (
            IQAwareController(RecordingController(), IQPerceptionEngine(), viewer_team="A"),
            PrivateInfoController(RecordingController(), viewer_team="A"),
        ):
            controller.set_game(game)
            view = controller.inner.games[-1]
            self.assertEqual(tuple(view.chars[1].pos), UNKNOWN_ENEMY_POS)
            self.assertEqual(view.replay_frames, ())

    def test_engine_reveals_only_targets_in_a_viewers_forward_vision(self):
        game = UltimateTestGame()
        attacker = make_character("attacker", "A", (3, 2))
        defender = make_character("defender", "D", (3, 5))
        game.chars = [attacker, defender]
        attacker.facing = defender.facing = "W"
        self.assertNotIn(defender.name, game._current_los_revealed_names())
        self.assertIn(attacker.name, game._current_los_revealed_names())
        attacker.facing = "E"
        self.assertIn(defender.name, game._current_los_revealed_names())
        attacker.blind_remaining = 1
        self.assertNotIn(defender.name, game._current_los_revealed_names())


if __name__ == "__main__":
    unittest.main()
