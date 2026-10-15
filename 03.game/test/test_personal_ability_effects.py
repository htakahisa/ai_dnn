"""発動演出ではゲームを進めず、演出終了後に元のtickを再開する。"""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock, patch

from effects.personal_ability_effects import load_personal_ability_effects
from test_match_playback import Scheduler
from test_ultimate_system import FixedController, make_character


SETTINGS = {"おもこ": {"追い詰められたおもこ": {
    "frames": [], "frame_interval_ms": 50, "duration_ms": 100,
    "sound": ["sound/ability_omoko.mp3", 0.5],
}}}


class PersonalAbilityEffectsTests(unittest.TestCase):
    def setUp(self):
        debug = patch("omoko_v1.ov1_ability_cornered_omoko.DEBUG_OMOKO_CORNERED_ENABLED", False)
        debug.start()
        self.addCleanup(debug.stop)
        settings = patch("effects.personal_ability_effects.load_personal_ability_effects", return_value=SETTINGS)
        settings.start()
        self.addCleanup(settings.stop)

    def game(self, *, headless=False):
        with redirect_stdout(StringIO()):
            from map_data import NEW_MAZE_STR
            from run_game import VisualFPSBattle, _build_team_ai
            game = VisualFPSBattle(NEW_MAZE_STR, _build_team_ai("default"),
                                   _build_team_ai("default"), headless=True,
                                   attacker_roster=["Leo", "Boaster", "Derke", "Chronicle", "Alfajer"],
                                   defender_roster=["Aspas", "valyn", "trent", "leaf", "tex"])
        actor = make_character("おもこ", "D", (10, 10))
        enemy = make_character("Demon1", "A", (20, 20))
        actor.accuracy = enemy.accuracy = 0
        game.chars = [actor, enemy]
        game.grid[18:23, 18:23] = 0
        game.defender_setup_phase.finish()
        game.attacker_controller = game.defender_controller = FixedController("MOVE")
        game.record_replay = False
        game.headless = headless
        game.root = Scheduler()
        game.draw = Mock()
        game.label = Mock()
        game._play_personal_ability_effect_sound = Mock()
        return game, actor

    def test_effect_freezes_entire_tick_then_resumes_ability_once(self):
        game, actor = self.game()
        before = (game.battle_tick, game.round_timer, game.detonate_timer, actor.hp, list(actor.pos))
        game.loop()
        duration = actor.omoko_cornered_remaining
        self.assertTrue(actor.omoko_cornered_triggered)
        self.assertEqual((game.battle_tick, game.round_timer, game.detonate_timer, actor.hp, actor.pos), before)
        self.assertFalse(game.step_tick())
        game.root.advance()  # 50ms: 次フレームのみ。
        self.assertEqual(actor.omoko_cornered_remaining, duration)
        self.assertEqual(game.battle_tick, 0)
        game.root.advance()  # 100ms: 元のtickを実行。
        self.assertIsNone(game.personal_ability_effect)
        self.assertEqual(game.battle_tick, 1)
        self.assertEqual(actor.omoko_cornered_remaining, duration - 1)
        self.assertNotEqual(actor.pos, before[-1])
        game._play_personal_ability_effect_sound.assert_called_once()
        game.root.advance()
        self.assertEqual(game.battle_tick, 2)
        self.assertIsNone(game.personal_ability_effect)

    def test_pause_and_resume_during_effect_preserves_tick_and_callback(self):
        game, _ = self.game()
        game.loop()
        game.toggle_pause()
        self.assertFalse(game.root.pending)
        elapsed = game.personal_ability_effect["animation_elapsed_ms"]
        game._advance_personal_ability_effect()
        self.assertEqual(game.personal_ability_effect["animation_elapsed_ms"], elapsed)
        self.assertEqual(game.battle_tick, 0)
        game.toggle_pause()
        self.assertEqual(next(iter(game.root.pending.values()))[1], game._advance_personal_ability_effect)
        game.root.advance()
        game.root.advance()
        self.assertEqual(game.battle_tick, 1)

    def test_debug_35_tick_trigger_waits_for_effect_before_tick_35(self):
        game, actor = self.game()
        game.chars.append(make_character("Leo", "D", (12, 12)))
        game.defender_setup_phase.state.ticks_elapsed = 30
        game.battle_tick = 3
        with patch("omoko_v1.ov1_ability_cornered_omoko.DEBUG_OMOKO_CORNERED_ENABLED", True):
            game.loop()  # 通算34tick。
            self.assertFalse(actor.omoko_cornered_triggered)
            self.assertEqual(game.battle_tick, 4)
            game.root.advance()  # 通算35tickの前に演出。
            self.assertIsNotNone(game.personal_ability_effect)
            self.assertEqual(game.battle_tick, 4)
            game.root.advance()
            self.assertEqual(game.battle_tick, 4)
            game.root.advance()
            self.assertIsNone(game.personal_ability_effect)
            self.assertEqual(game.battle_tick, 5)

    def test_multiple_effects_play_in_order_without_intervening_game_tick(self):
        game, actor = self.game()
        game.loop()
        game._queue_personal_ability_effect(actor, "追い詰められたおもこ")
        for _ in range(3):
            game.root.advance()
            self.assertEqual(game.battle_tick, 0)
        game.root.advance()
        self.assertEqual(game.battle_tick, 1)
        self.assertEqual(game._play_personal_ability_effect_sound.call_count, 2)

    def test_headless_and_unconfigured_effects_never_pause_gameplay(self):
        for headless in (True, False):
            with self.subTest(headless=headless):
                game, actor = self.game(headless=headless)
                with patch("effects.personal_ability_effects.load_personal_ability_effects", return_value={}):
                    self.assertTrue(game.step_tick())
                self.assertTrue(actor.omoko_cornered_triggered)
                self.assertEqual(game.battle_tick, 1)
                self.assertIsNone(game.personal_ability_effect)
                game._play_personal_ability_effect_sound.assert_not_called()

    def test_fast_forward_clears_effect_and_continues_without_animation(self):
        game, _ = self.game()
        game.loop()
        game.start_fast_forward("round")
        self.assertIsNone(game.personal_ability_effect)
        self.assertEqual(game.personal_ability_effect_queue, [])
        self.assertEqual(next(iter(game.root.pending.values()))[1], game._run_fast_forward_chunk)
        game.headless = True
        self.assertTrue(game.step_tick())
        self.assertEqual(game.battle_tick, 1)

    def test_round_initialization_clears_effect_state(self):
        game, _ = self.game()
        game.loop()
        game.headless = True
        with redirect_stdout(StringIO()):
            game.init_round()
        self.assertIsNone(game.personal_ability_effect)
        self.assertEqual(game.personal_ability_effect_queue, [])

    def test_shared_banner_uses_ability_name_with_or_without_images(self):
        from rendering_ui import RenderingUIMixin
        game, _ = self.game()
        game.loop()
        game.canvas = Mock()
        image = Mock()
        image.width.return_value = image.height.return_value = 200
        for images in ([], [image]):
            with patch.object(game, "_get_ace_effect_frames", return_value=images):
                game.canvas.reset_mock()
                RenderingUIMixin._draw_character_effect_banner(game, game.personal_ability_effect)
                texts = [call.kwargs.get("text") for call in game.canvas.create_text.call_args_list]
                self.assertIn("追い詰められたおもこ", texts)
                self.assertNotIn("ACE!", texts)
                self.assertEqual(game.canvas.create_image.called, bool(images))

    def test_default_configuration_uses_requested_asset_directories(self):
        # ローダーのモック外で実ファイルを確認する。
        settings = load_personal_ability_effects()
        effect = settings["おもこ"]["追い詰められたおもこ"]
        self.assertEqual(effect["frames"], ["image/ability_omoko01.png", "image/ability_omoko02.png"])
        self.assertEqual(effect["sound"][0], "sound/ability_omoko.mp3")


if __name__ == "__main__":
    unittest.main()
