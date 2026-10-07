"""Saved player details, power sorting, and the home payroll forecast."""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import character_stats
from character_stats import CharacterStats
from realtime_season import SeasonSaveError, player_from_save, validate_player
from rendering_ui import RenderingUIMixin
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

    def test_awakening_description_does_not_affect_combat_power_or_numeric_validation(self):
        player = CharacterStats("Example", .5, .25, 180, .8, 110, "スモーカー", 50)
        described = replace(player, awakening_description="移動速度上昇\nアビリティ回復")
        validate_player(described)
        self.assertEqual(player_combat_power(described), player_combat_power(player))
        self.assertEqual(player_duel_power(described), player_duel_power(player))
        with self.assertRaisesRegex(SeasonSaveError, "覚醒説明"):
            validate_player(replace(player, awakening_description=123))

    def test_custom_awakening_announcement_preserves_combo_text_and_blank_fallback(self):
        player = replace(character_stats.get_by_name("Leo"), awakening_description="移動速度上昇とアビリティ回復")
        event = dict(type="awakening", players=("Leo",), display_players=("-ラスボス-Leo",), effect_text="旧説明")
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": player}):
            self.assertEqual(RenderingUIMixin._announcement_effect_text(event), player.awakening_description)
            self.assertEqual(RenderingUIMixin._announcement_effect_text({**event, "type": "combo"}), "旧説明")
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": replace(player, awakening_description="   ")}):
            self.assertEqual(RenderingUIMixin._announcement_effect_text(event), "旧説明")
        self.assertEqual(RenderingUIMixin._announcement_effect_text({**event, "players": ("Unknown",)}), "旧説明")


class PlayerDetailsScreenTest(SeasonEconomyScreenTest):
    def test_catalog_description_reaches_editor_scout_and_initial_selection_with_long_text(self):
        description = "\n".join(f"覚醒効果 {n}: 移動速度上昇とアビリティ回復" for n in range(30))
        player = replace(character_stats.get_by_name("Leo"), awakening_description=description)
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": player}):
            app = self.app
            app.show_editor()
            app.players.selection_set("0")
            app.select_player()
            self.assertIn(description, app.details.get())
            self.assertIn(description, app.player_details_view.text.get("1.0", "end"))
            self.assertEqual(str(app.player_details_view.text["state"]), "disabled")
            app.root.update_idletasks()
            self.assertLessEqual(app.editor_host.winfo_reqheight(), 800)
            app.player_details_view.text.yview_moveto(1)
            self.assertAlmostEqual(app.player_details_view.text.yview()[1], 1)
            app.show_screen("scout")
            app.scout_filter.set("全選手")
            app.scout_players.selection_set("Leo")
            app.refresh_offer("scout")
            self.assertIn(description, app.scout_details_view.text.get("1.0", "end"))
            app.root.update_idletasks()
            self.assertLessEqual(app.scout_host.winfo_reqheight(), 800)
            app.scout_players.selection_set("Meiy")
            app.refresh_offer("scout")
            self.assertNotIn(description, app.scout_details_view.text.get("1.0", "end"))
            app.refresh_starters()
            app.starter_players.selection_set("Leo")
            app.preview_starter()
            self.assertIn(description, app.starter_details_view.text.get("1.0", "end"))

    def test_legacy_save_gets_catalog_description_and_later_edits_do_not_reset_growth(self):
        state = self.app.state
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        for row in data["owned_players"]:
            row.pop("awakening_description", None)
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        original = self.store.path.read_bytes()
        player = replace(character_stats.get_by_name("Leo"), awakening_description="リコンが回復します。")
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": player}):
            loaded = self.store.load_or_create()
            self.assertEqual(loaded.player("Leo").awakening_description, player.awakening_description)
            self.assertEqual(loaded.player("Leo").iq, state.player("Leo").iq)
            self.assertEqual(loaded.contracts, state.contracts)
            self.assertEqual(self.store.path.read_bytes(), original)
            self.store.save(loaded)
            self.assertEqual(self.store.load_or_create(), loaded)
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": replace(player, awakening_description="新しい説明")}):
            self.assertEqual(character_stats.get_awakening_description(loaded.player("Leo")), "新しい説明")
        unknown = player_from_save(dict(name="Unknown", hs_pct=.3, dodge_pct=.1, iq=100,
                                      hit_pct=.7, reaction=100, role="シーカー", influence=50))
        self.assertEqual(unknown.awakening_description, "")

    def test_all_stat_fields_use_saved_player_and_match_editor_power(self):
        app = self.app
        changed = replace(app.state.player("Leo"), iq=400, hit_pct=.91, influence=157, mental=9,
                          form_variance=3, shield_hp=44.5, shield_piercer=True, shield_crash=21.25)
        state = replace(app.state, owned_players=tuple(changed if p.name == "Leo" else p for p in app.state.owned_players))
        app.commit(state, "Saved abilities")
        app.show_screen("scout")
        app.scout_filter.set("全選手")
        app.scout_players.selection_set("Leo")
        app.refresh_offer("scout")
        details = app.scout_details.get()
        for label in ("Leo", changed.role, "HS率", "命中率", "回避率", "反応", "IQ", "影響力",
                      "メンタル", "調子の波", "基本月給", "忠誠心", "チームへの忠誠",
                      "シールド", "シールドピアサー", "シールドクラッシュ"):
            self.assertIn(label, details)
        self.assertIn("IQ: 400", details)
        self.assertIn("影響力: 157", details)
        self.assertIn("91.0%", details)
        app.show_editor()
        app.players.selection_set("0")
        app.select_player()
        for shield_value in ("シールド: 44.5HP", "シールドピアサー: あり", "シールドクラッシュ: 21.25HP"):
            self.assertIn(shield_value, details)
            self.assertIn(shield_value, app.details.get())
            self.assertIn(shield_value, app.player_details_view.text.get("1.0", "end"))
        for kind in ("research", "aim_lab"):
            app.training_players[kind].selection_set("Leo")
            app.refresh_training_offer(kind)
            for shield_value in ("シールド: 44.5HP", "シールドピアサー: あり", "シールドクラッシュ: 21.25HP"):
                self.assertIn(shield_value, app.training_summaries[kind].get())
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
        self.assertEqual(app.state.monthly_payroll, 90_000)
        self.assertIn("-90,000", app.home_payroll.get())


def load_tests(loader, tests, pattern):
    suite = loader.loadTestsFromTestCase(PowerIndexTest)
    for name in PlayerDetailsScreenTest.__dict__:
        if name.startswith("test_"):
            suite.addTest(PlayerDetailsScreenTest(name))
    return suite


if __name__ == "__main__":
    unittest.main()
