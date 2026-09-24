from pathlib import Path
import re
import unittest

from map_data import NEW_MAZE_STR
from party_presets import PARTY_PRESETS


ROOT = Path(__file__).resolve().parent
DESIGN_PATH = ROOT / "03.DESIGN.md"


class CoachV1Task00DesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = DESIGN_PATH.read_text(encoding="utf-8")

    def test_fixed_map_contract_matches_current_map(self):
        rows = [row for row in NEW_MAZE_STR.splitlines() if row]
        self.assertEqual(26, len(rows))
        self.assertTrue(rows)
        self.assertEqual({44}, {len(row) for row in rows})
        self.assertIn("26 行 × 44 列", self.design)

    def test_locked_roster_matches_existing_preset_order(self):
        expected = ("ごりまる", "ごんごん", "ごんた", "くんた", "くりまる")
        self.assertEqual(expected, PARTY_PRESETS["Gorigons"].players)

        roster_section = self.design.split("## 4.", 1)[0]
        documented = re.findall(
            r"^\| [0-4] \| ([^|]+?) \|", roster_section, flags=re.MULTILINE
        )
        self.assertEqual(list(expected), documented)

    def test_tiger_has_no_normal_active_ability(self):
        self.assertIn("HUNT（通常active abilityなし）", self.design)
        self.assertIn("ごんごんの通常ability使用actionは常にmask", self.design)

    def test_action_ownership_and_priority_are_explicit(self):
        required = (
            "coachだけが次を決める",
            "キャラクター固有モデルだけが次を決める",
            "PLANTとDEFUSEは完了まで毎tick coachが再選択",
            "ULTIMATEの選択・実行",
            "オーブ取得の選択・実行",
            "半角90度、合計180度",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.design)

    def test_actor_contract_rejects_hidden_enemy_truth_sources(self):
        forbidden_sources = (
            'game_state["chars"]',
            "PerceivedGameView.real_game",
            "PerceivedCharacter.real_character",
            "未視認敵の現在またはぼかした現在座標",
            "critic用の敵実座標",
        )
        for source in forbidden_sources:
            with self.subTest(source=source):
                self.assertIn(source, self.design)

        self.assertIn(
            "未視認敵の実位置だけが異なる二状態から、\n同一のactor観測",
            self.design,
        )

    def test_versions_and_checkpoint_split_are_frozen(self):
        required = (
            "coach-checkpoint-v1",
            "coach-observation-v1",
            "coach-action-v1",
            "character-observation-v1",
            "character-action-v1",
            "checkpoints/coach/attacker/best.pt",
            "checkpoints/coach/defender/best.pt",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.design)

    def test_task00_requires_no_core_change(self):
        self.assertIn("Task 00ではcore変更は不要であり、変更しない", self.design)
        self.assertIn("team_ai.py", self.design)


if __name__ == "__main__":
    unittest.main()
