"""Saved player details, power sorting, and the home payroll forecast."""

from dataclasses import replace
import unittest

from character_stats import CharacterStats
from season.season_player_stats import player_combat_power, player_duel_power
from test_season_economy import OWN, SeasonEconomyScreenTest


class PowerIndexTest(unittest.TestCase):
    def test_rate_scale_and_removing_iq_contribution(self):
        player = CharacterStats("Example", .5, .25, 180, .8, 110, "スモーカー", 50)
        self.assertAlmostEqual(player_combat_power(player), 325.5)
        self.assertAlmostEqual(player_duel_power(player), 235.5)
        raised_iq = replace(player, iq=400)
        self.assertAlmostEqual(player_combat_power(raised_iq), 435.5)
        self.assertEqual(player_duel_power(raised_iq), player_duel_power(player))


class PlayerDetailsScreenTest(SeasonEconomyScreenTest):
    def test_all_stat_fields_use_saved_player_and_match_editor_power(self):
        app = self.app
        changed = replace(app.state.player("Leo"), iq=400, hit_pct=.91, influence=157, mental=9, form_variance=3)
        state = replace(app.state, owned_players=tuple(changed if p.name == "Leo" else p for p in app.state.owned_players))
        app.commit(state, "Saved abilities")
        app.show_screen("scout")
        app.scout_filter.set("全選手")
        app.scout_players.selection_set("Leo")
        app.refresh_offer("scout")
        details = app.scout_details.get()
        for label in ("Leo", changed.role, "HS率", "命中率", "回避率", "反応", "IQ", "影響力",
                      "メンタル", "調子の波", "基本月給", "忠誠心", "チームへの忠誠"):
            self.assertIn(label, details)
        self.assertIn("IQ: 400", details)
        self.assertIn("影響力: 157", details)
        self.assertIn("91.0%", details)
        app.show_editor()
        app.players.selection_set("0")
        app.select_player()
        for power in (f"総合戦闘力: {player_combat_power(changed):.2f}",
                      f"撃ち合い戦闘力: {player_duel_power(changed):.2f}"):
            self.assertIn(power, details)
            self.assertIn(power, app.details.get())
        app.scout_filter.set("LFTのみ")
        self.assertEqual(app.scout_details.get(), "選手を選択すると全ステータスを表示します。")

    def test_sort_metrics_and_directions_keep_unavailable_players_last_and_selection(self):
        app = self.app
        app.show_screen("scout")
        app.scout_filter.set("全選手")
        app.commit(replace(app.state, money=1_600_000), "LFT affordable, transfers unaffordable")
        app.scout_players.selection_set("Meiy")
        app.refresh_offer("scout")
        players = {p.name: p for p in app.state.scout_players}
        metrics = {"IQ": lambda p: p.iq, "総合戦闘力": player_combat_power, "撃ち合い戦闘力": player_duel_power}
        for label, metric in metrics.items():
            for order, sign in (("高い順", -1), ("低い順", 1)):
                with self.subTest(metric=label, order=order):
                    app.scout_sort.set(label)
                    app.scout_sort_order.set(order)
                    names = app.scout_players.get_children()
                    unavailable = lambda name: "contract_unavailable" in app.scout_players.item(name, "tags")
                    expected = sorted(names, key=lambda name: (unavailable(name), sign * metric(players[name]), name.casefold()))
                    self.assertEqual(list(names), expected)
                    self.assertTrue(unavailable("Leo"))
                    self.assertTrue(unavailable("Sato"))
                    self.assertFalse(unavailable("Meiy"))
                    self.assertEqual(app.scout_players.selection(), ("Meiy",))
        app.change_scout_sort("IQ")
        self.assertEqual(app.scout_sort.get(), "IQ")
        self.assertEqual(app.scout_sort_order.get(), "高い順")
        app.change_scout_sort("IQ")
        self.assertEqual(app.scout_sort_order.get(), "低い順")
        app.scout_search.set("no matching player")
        self.assertFalse(app.scout_players.get_children())
        self.assertEqual(app.scout_details.get(), "選手を選択すると全ステータスを表示します。")

    def test_home_payroll_is_red_and_counts_contracts_once_including_discount(self):
        app = self.app
        self.assertEqual(app.state.monthly_payroll, 500_000)
        self.assertIn("-500,000", app.home_payroll.get())
        self.assertEqual(str(app.home_payroll_label["foreground"]), "#c62828")
        state = app.state.with_new_team().with_roster(OWN).with_confirmed_team()
        self.assertEqual(state.monthly_payroll, 500_000)
        state = state.with_scouted_player("Meiy", "year2")
        app.commit(state, "Discounted salary")
        self.assertEqual(app.state.monthly_payroll, 590_000)
        self.assertIn("-590,000", app.home_payroll.get())
        expired = state.advance_months(12, pay_salaries=False)
        app.commit(expired, "Initial contracts expired")
        self.assertEqual(app.state.monthly_payroll, 90_000)
        self.assertIn("-90,000", app.home_payroll.get())
        app.commit(app.state.without_player("Meiy"), "Released remaining active player")
        self.assertEqual(app.state.monthly_payroll, 0)


def load_tests(loader, tests, pattern):
    suite = loader.loadTestsFromTestCase(PowerIndexTest)
    for name in PlayerDetailsScreenTest.__dict__:
        if name.startswith("test_"):
            suite.addTest(PlayerDetailsScreenTest(name))
    return suite


if __name__ == "__main__":
    unittest.main()
