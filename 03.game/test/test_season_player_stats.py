from dataclasses import asdict, replace
import unittest

from character_stats import CHARACTER_TABLE, CharacterStats
from realtime_season import SeasonSaveError, player_from_save, validate_player
from season.season_player_stats import SeasonPlayerStats, season_player


class SeasonPlayerStatsTests(unittest.TestCase):
    def player(self):
        return CharacterStats("Example", .3, .2, 100, .7, 100, "フラッシュ", 50,
                              5, 5, 0, False, 0, 100_000, 5, 2)

    def test_catalog_has_no_training_levels_and_positional_chapter_is_correct(self):
        self.assertEqual(self.player().debut_chapter, 2)
        for player in (*CHARACTER_TABLE.values(), self.player()):
            self.assertNotIn("research_level", asdict(player))
            self.assertNotIn("aim_lab_level", asdict(player))

    def test_season_players_start_at_zero_and_keep_catalog_abilities(self):
        catalog = self.player()
        player = season_player(catalog)
        self.assertIsInstance(player, SeasonPlayerStats)
        self.assertEqual((player.research_level, player.aim_lab_level), (0, 0))
        self.assertEqual(asdict(catalog), {key: value for key, value in asdict(player).items()
                                         if key not in {"research_level", "aim_lab_level"}})
        trained = replace(player, research_level=3, aim_lab_level=4, iq=115, hit_pct=.78)
        self.assertIs(season_player(trained), trained)
        self.assertEqual(catalog.iq, 100)

    def test_old_saved_training_levels_and_abilities_are_preserved(self):
        row = dict(asdict(self.player()), research_level=7, aim_lab_level=12,
                   iq=135, hit_pct=.94)
        loaded = player_from_save(row)
        validate_player(loaded)
        self.assertEqual(asdict(loaded), row)
        self.assertEqual(player_from_save(asdict(self.player())), season_player(self.player()))

    def test_season_training_levels_are_still_validated(self):
        for field in ("research_level", "aim_lab_level"):
            for value in (-1, 31, 1.5, True):
                with self.subTest(field=field, value=value), self.assertRaises(SeasonSaveError):
                    validate_player(replace(season_player(self.player()), **{field: value}))


if __name__ == "__main__":
    unittest.main()
