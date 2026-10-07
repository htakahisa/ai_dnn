"""Expired contracts are excluded from editable lineups and guide renewal."""

from dataclasses import replace
import tkinter as tk
import unittest
from unittest.mock import patch

from realtime_season import EXPIRED_ROSTER_WARNING, SeasonSaveError
from run_realtime_season import RealtimeSeasonApp
import test_season_tournament_contracts as fixtures
from test_season_competitions import OWN


class ExpiredLineupTests(unittest.TestCase):
    setUp = fixtures.TournamentContractTests.setUp
    state = fixtures.TournamentContractTests.state
    in_progress = fixtures.TournamentContractTests.in_progress

    def expired(self):
        state = self.state().with_preset_settings(igl="Leo", carrier="Leo", ai="fnatic_v3")
        return state.advance_days(31)

    def app(self, state):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        self.store.save(state)
        return RealtimeSeasonApp(root, self.store, state)

    def select(self, app, name):
        index = next(i for i, p in enumerate(app.state.owned_players) if p.name == name)
        app.players.selection_set(str(index))
        app.select_player()
        return str(index)

    def test_core_blocks_expired_addition_and_confirmation_but_legacy_save_loads(self):
        state = self.expired()
        self.assertIn("Leo", state.roster)
        self.assertFalse(state.can_play("Leo"))
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)
        with self.assertRaisesRegex(SeasonSaveError, EXPIRED_ROSTER_WARNING):
            state.with_roster(OWN)
        with self.assertRaisesRegex(SeasonSaveError, EXPIRED_ROSTER_WARNING):
            state.with_confirmed_team()

    def test_opening_existing_save_repairs_draft_and_roles_without_changing_contract(self):
        state = self.expired()
        app = self.app(state)
        self.assertEqual(app.state.roster, OWN[1:])
        self.assertIsNone(app.state.preset_igl)
        self.assertIsNone(app.state.preset_carrier)
        self.assertEqual(app.state.preset_ai, "fnatic_v3")
        self.assertEqual(app.state.contract("Leo"), state.contract("Leo"))
        self.assertEqual(app.state.owned_players, state.owned_players)
        self.assertEqual(app.state.game_date, state.game_date)
        self.assertEqual(app.state.money, state.money)
        self.assertEqual(app.state.teams, state.teams)
        self.assertEqual(self.store.load_or_create(), app.state)
        self.assertIn(EXPIRED_ROSTER_WARNING, app.status.get())
        self.assertIn("再契約", app.editor_contract_warning.get())
        self.assertFalse(app.state.roster_ready)
        self.assertEqual(str(app.confirm_button["state"]), "disabled")

    def test_expired_inventory_is_red_and_cannot_be_added_by_button_or_double_click(self):
        app = self.app(self.expired())
        app.show_editor()
        identifier = self.select(app, "Leo")
        self.assertIn("contract_expired", app.players.item(identifier, "tags"))
        self.assertIn("契約終了", app.players.item(identifier, "values")[-1])
        self.assertEqual(str(app.add_button["state"]), "disabled")
        self.assertIn(EXPIRED_ROSTER_WARNING, app.details.get())
        before = app.state
        app.add_player()
        self.assertEqual(app.state, before)
        self.assertNotIn("Leo", app.state.roster)
        self.assertIn(EXPIRED_ROSTER_WARNING, app.status.get())

    def test_renewal_button_selects_contract_and_renewal_allows_restoring_preset(self):
        app = self.app(self.expired())
        original = app.state.teams[0]
        self.select(app, "Leo")
        app.editor_renew_button.invoke()
        self.assertEqual(app.current_screen, "contracts")
        self.assertEqual(app.contracts_players.selection(), ("Leo",))
        self.assertEqual(str(app.offer_buttons["contracts"]["state"]), "normal")
        app.sign_selected_contract("contracts")
        self.assertTrue(app.state.can_play("Leo"))
        self.assertNotIn("Leo", app.state.roster)
        app.show_editor()
        identifier = self.select(app, "Leo")
        self.assertNotIn("contract_expired", app.players.item(identifier, "tags"))
        self.assertEqual(str(app.add_button["state"]), "normal")
        app.add_player()
        app.confirm()
        self.assertEqual(len(app.state.teams), 1)
        self.assertEqual(app.state.teams[0].id, original.id)
        self.assertIn("Leo", app.state.teams[0].roster)
        self.assertTrue(all(app.state.can_play(n) for n in app.state.teams[0].roster))
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_loading_old_preset_cannot_reintroduce_expired_members(self):
        app = self.app(self.expired())
        app.new_team()
        app.edit_team_choice.set(app.state.teams[0].name)
        app.edit_saved_team()
        self.assertEqual(app.state.roster, OWN[1:])
        self.assertIn("再契約", app.status.get())
        self.assertEqual(self.store.load_or_create().roster, OWN[1:])

    def test_contract_expiry_during_calendar_action_cleans_roster_in_same_save(self):
        state = self.state().advance_days(29)
        self.assertTrue(state.can_play("Leo"))
        app = self.app(state)
        app.advance_calendar(1)
        self.assertEqual(app.state.game_date, "2026-01-31")
        self.assertNotIn("Leo", app.state.roster)
        self.assertIn("Leo", app.editor_contract_warning.get())
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_tournament_contract_extension_keeps_expired_starter_usable(self):
        state = self.in_progress()
        self.assertTrue(state.contract_end_deferred(state.contract("Leo")))
        app = self.app(state)
        self.assertIn("Leo", app.state.roster)
        self.assertEqual(app.state, state)
        identifier = self.select(app, "Leo")
        self.assertNotIn("contract_expired", app.players.item(identifier, "tags"))
        self.assertNotIn("Leo", app.editor_contract_warning.get())
        self.assertIn("Meiy", app.editor_contract_warning.get())
        app.state.with_confirmed_team()

    def test_failed_repair_save_keeps_original_data_and_blocks_confirm(self):
        state = self.expired()
        self.store.save(state)
        original = self.store.path.read_bytes()
        with patch.object(self.store, "save", side_effect=OSError("disk unavailable")), \
                patch("run_realtime_season.messagebox.showerror"):
            root = tk.Tk()
            root.withdraw()
            self.addCleanup(root.destroy)
            app = RealtimeSeasonApp(root, self.store, state)
        self.assertEqual(self.store.path.read_bytes(), original)
        self.assertEqual(app.state, state)
        self.assertEqual(str(app.confirm_button["state"]), "disabled")
        self.assertNotIn("編成完了", app.summary.get())
        app.refresh()
        self.assertNotIn("Leo", app.state.roster)
        self.assertEqual(self.store.load_or_create(), app.state)


if __name__ == "__main__":
    unittest.main()
