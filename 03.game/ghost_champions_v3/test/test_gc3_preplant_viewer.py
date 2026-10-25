"""Viewer model selection and inference phase labels."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ghost_champions_v3.tools.gc3_watch_attacker_preplant import AttackerBattle, prepare_round, MODEL_PATHS


class Status:
    def set(self, value):
        self.value = value


class PreplantViewerTests(unittest.TestCase):
    def test_latest_best_and_bc_are_explicit_distinct_choices(self):
        self.assertEqual(MODEL_PATHS["RL最新"].name,"latest.pt")
        self.assertEqual(MODEL_PATHS["RL best"].name,"best.pt")
        self.assertNotEqual(MODEL_PATHS["RL best"].parent,MODEL_PATHS["BC best"].parent)

    def test_unknown_checkpoint_choice_is_rejected_without_fresh_fallback(self):
        with self.assertRaisesRegex(ValueError,"Unknown model"):
            prepare_round("missing-model","FNC",1)

    def test_actual_plant_selects_postplant_label(self):
        game=AttackerBattle.__new__(AttackerBattle)
        game.viewer_status=Status()
        game.viewer_description="RL latest 80043"
        game.defender_setup_phase=SimpleNamespace(active=False)
        game.is_planted=True
        game.battle_tick=40
        game.round_over=False
        with patch("ghost_champions_v3.tools.gc3_watch_attacker_postplant.PostplantBattle.draw",return_value=None):
            AttackerBattle.draw(game)
        self.assertIn("設置後",game.viewer_status.value)

    def test_sampled_goal_site_is_visible_with_the_new_model(self):
        game=AttackerBattle.__new__(AttackerBattle)
        game.viewer_status=Status()
        game.viewer_description="Omoko BC"
        game.defender_setup_phase=SimpleNamespace(active=False)
        game.is_planted=False;game.battle_tick=1;game.round_over=False
        history=SimpleNamespace(round_id=2,goal_sites={2:1})
        game.attacker_controller=SimpleNamespace(inner_controller=SimpleNamespace(preplant=SimpleNamespace(history=history)))
        with patch("ghost_champions_v3.tools.gc3_watch_attacker_postplant.PostplantBattle.draw",return_value=None):
            AttackerBattle.draw(game)
        self.assertIn("目標サイト=B",game.viewer_status.value)


if __name__=="__main__":
    unittest.main()
