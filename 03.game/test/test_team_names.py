import unittest
from types import SimpleNamespace
from party_presets import PARTY_PRESETS, canonical_preset_name, get_preset, get_team_short_name, normalize_team_names


class TeamNameTests(unittest.TestCase):
    def test_aliases(self):
        pairs = [
            ("とうやまゲーミング", "Touyama Gaming"),
            ("ドラゴンテイル", "Dragon Tail"),
            ("ブラッドムウン", "Blood Moon"),
            ("Leo軸", "Leo And Friends"),
            ("フリーナクラシック", "Furina Classic"),
            ("フリーナタルタリヤ", "Furina Tartaglia"),
            ("日本代表", "Japan All-Stars"),
            ("クイーンズフラワーギャンビット", "Queen's Flower Gambit"),
            ("アイネクライネ", "Eine Kleine"),
            ("個人能力パ", "Team Elites"),
        ]
        for old, new in pairs:
            self.assertEqual(canonical_preset_name(old), new)
            self.assertIs(get_preset(old), get_preset(new))
            self.assertEqual(get_preset(new).name, new)

    def test_short_names(self):
        names = [p.short_name for p in PARTY_PRESETS.values()]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(names))
        self.assertEqual(get_team_short_name('Touyama Gaming'), 'TYG')
        self.assertEqual(get_team_short_name('unknown'), 'unknown')

    def test_nested_history(self):
        old = {'team1': 'とうやまゲーミング', 'score': 13,
               'players': {'夢の街': {'team': 'とうやまゲーミング', 'kills': 20}},
               'sides': {'とうやまゲーミング': {'wins': 8}}}
        new = normalize_team_names(old)
        self.assertEqual(new['team1'], 'Touyama Gaming')
        self.assertEqual(new['players']['夢の街']['kills'], 20)
        self.assertEqual(new['sides']['Touyama Gaming']['wins'], 8)
        self.assertEqual(old['team1'], 'とうやまゲーミング')

    def test_checkpoint_slots(self):
        from gc_v1.learning_defender_setup_gc_runtime import _build_obs, PLAYER_COUNT, ACTION_DIM, OPPONENT_DIM
        offset = PLAYER_COUNT + ACTION_DIM + 2 + ACTION_DIM * 2
        obs = _build_obs(SimpleNamespace(pos=[1, 1]), 0, [],
                         'とうやまゲーミング', 0,
                         {'Touyama Gaming': OPPONENT_DIM - 1})
        self.assertEqual(obs[offset:offset + OPPONENT_DIM].sum(), 1)
        self.assertEqual(obs[offset + OPPONENT_DIM - 1], 1)


if __name__ == '__main__':
    unittest.main()
