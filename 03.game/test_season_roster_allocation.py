"""Initial ownership allocation and replacement tournament invitations."""

from dataclasses import replace
from pathlib import Path
from random import Random
import json
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_config as settings
import realtime_season_rival_economy as economy
import realtime_season_teams as teams
from realtime_season import SeasonStore, new_season
from season_competitions import definition_from_dict
from test_season_competitions import OWN, RIVALS, definition, finish


class RosterAllocationTests(unittest.TestCase):
    def setUp(self):
        fixture = {n: replace(p, monthly_salary=100_000, loyalty=10)
                   for n, p in character_stats.CHARACTER_TABLE.items()}
        self.clubs = [dict(name=f"Rival{i}", players=list(names), initial_money=100_000_000)
                      for i, names in enumerate(RIVALS)]
        for context in (
            patch.dict(character_stats.CHARACTER_TABLE, fixture),
            patch.object(calendar, "START_DATE", "2026-01-01"),
            patch.object(calendar, "TOURNAMENTS", []),
            patch.object(settings, "INITIAL_OWNED_PLAYERS", (*OWN, "Meiy", "Less")),
            patch.object(teams, "SEASON_TEAMS", self.clubs),
            patch.object(economy, "NON_REGULAR_OFFER_CHANCE", 0),
        ):
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self):
        state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        return state.with_selected_team(state.teams[0].id)

    def test_chosen_starters_override_every_rival_without_cost_or_day(self):
        for club in self.clubs:
            club["players"] = [*OWN, *club["players"]]
            club.update(igl="Leo", carrier="Boaster")
        pending = new_season()
        started = pending.with_initial_selection(OWN)
        self.assertEqual(tuple(p.name for p in started.owned_players), OWN)
        self.assertEqual(started.date, pending.date)
        self.assertEqual(started.money, pending.money)
        for club in started.opponent_teams:
            self.assertTrue(set(OWN).isdisjoint(club.members))
            self.assertTrue(set(OWN).isdisjoint(club.regular_members))
            self.assertNotIn(club.igl, OWN)
            self.assertNotIn(club.carrier, OWN)
            self.assertTrue(all(c.end_reason == "released" for c in club.contracts if c.player_name in OWN))
        started.validate()

    def test_explicit_initial_distribution_also_has_priority(self):
        self.clubs[0]["players"] = [*OWN, *RIVALS[0]]
        state = self.state()
        self.assertTrue(set(OWN).isdisjoint(state.opponent_teams[0].members))
        self.assertEqual(tuple(p.name for p in state.owned_players), OWN)

    def test_shared_player_randomly_chooses_one_listed_club_and_stays_there(self):
        self.clubs[1]["players"] = ["Aspas", *RIVALS[1]]
        owners = set()
        for seed in range(12):
            with patch("season_transfers.Random", return_value=Random(seed)):
                pending = new_season()
            owner = pending.opponent_owner("Aspas").id
            started = pending.with_initial_selection(OWN)
            self.assertEqual(started.opponent_owner("Aspas").id, owner)
            self.assertEqual(sum("Aspas" in c.members for c in started.opponent_teams), 1)
            self.assertEqual(sum("Aspas" in c.regular_members for c in started.opponent_teams), 1)
            self.assertIn("Aspas", started.opponent_owner("Aspas").initial_shared_members)
            started.validate()
            owners.add(owner)
        self.assertEqual(len(owners), 2)

    def test_pending_save_load_and_import_keep_the_allocated_owner(self):
        self.clubs[1]["players"] = ["Aspas", *RIVALS[1]]
        pending = new_season()
        self.store.save(pending)
        with patch("season_transfers.Random", side_effect=AssertionError("allocation repeated")):
            loaded = self.store.load_or_create()
            self.assertEqual(loaded, pending)
            imported = self.store.import_season_teams(loaded)
            self.assertEqual(imported.opponent_owner("Aspas").id, pending.opponent_owner("Aspas").id)
        started = loaded.with_initial_selection(OWN)
        self.store.save(started)
        self.assertEqual(self.store.load_or_create(), started)

    def test_short_and_empty_regular_rosters_remain_valid_through_monthly_update(self):
        for club in self.clubs:
            club["players"] = list(OWN)
        state = new_season().with_initial_selection(OWN)
        self.assertTrue(all(not c.players for c in state.opponent_teams))
        state.validate()
        updated = state.advance_days(31)
        self.assertTrue(all(len(c.players) == 5 for c in updated.opponent_teams))
        updated.validate()

    def test_losing_club_does_not_try_to_reclaim_shared_regular(self):
        self.clubs[1]["players"] = ["Aspas", *RIVALS[1]]
        state = new_season().with_initial_selection(OWN)
        owner = state.opponent_owner("Aspas").id
        updated = state.advance_days(31)
        self.assertEqual(updated.opponent_owner("Aspas").id, owner)
        self.assertEqual(sum("Aspas" in c.regular_members for c in updated.opponent_teams), 1)

    def test_unselected_candidate_is_not_removed_for_player_priority(self):
        self.clubs[0]["players"].append("Meiy")
        state = new_season().with_initial_selection(OWN)
        self.assertIsNone(state.player("Meiy"))
        self.assertIsNotNone(state.opponent_owner("Meiy"))

    def test_tournament_replaces_short_named_invitees_with_other_full_teams(self):
        self.clubs.append(dict(name="Backup", players=["Laz", "Xdll", "SyouTa", "Absol", "SugarZ3ro"]))
        self.clubs[0]["players"] = list(RIVALS[0][:4])
        cup = definition_from_dict(definition(start_date="2026-01-05", opponent_teams=("Rival0", "Rival1", "Rival2")))
        state = replace(self.state(), tournament_definitions=(cup,))
        registered = state.with_tournament_entry("cup", state.selected_team_id)
        run = registered.tournament("cup")
        self.assertEqual([t.name for t in run.entrants[1:]], ["Rival1", "Rival2", "Backup"])
        self.assertEqual(len(run.entrants), 4)
        self.assertTrue(all(len(t.players) == 5 for t in run.entrants))
        self.assertEqual(state.opponent_teams, registered.opponent_teams)

    def test_npc_calendar_uses_the_same_replacement_pool(self):
        self.clubs[0]["players"] = list(RIVALS[0][:4])
        cup = definition_from_dict(definition(start_date="2026-01-05", team_count=2,
                     format="single_elimination", prizes={}, opponent_teams=("Rival0", "Rival1")))
        state = replace(self.state(), tournament_definitions=(cup,)).advance_days(5)
        self.assertEqual([t.name for t in state.tournament("cup").entrants], ["Rival1", "Rival2"])
        self.assertTrue(state.tournament("cup").completed)

    def test_shortage_uses_available_field_and_saves_real_results_and_prizes(self):
        self.clubs[0]["players"] = list(RIVALS[0][:4])
        cup = definition_from_dict(definition(start_date="2026-01-05"))
        state = replace(self.state(), tournament_definitions=(cup,))
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(4)
        self.assertEqual(len(state.tournament("cup").entrants), 3)
        completed = finish(state)
        self.assertEqual(len(completed.tournament("cup").results), 4)
        self.assertEqual(completed.tournament("cup").prize_paid, cup.prizes[1])
        self.store.save(completed)
        self.assertEqual(self.store.load_or_create(), completed)

    def test_no_available_opponent_waits_without_creating_invalid_progress(self):
        for club in self.clubs:
            club["players"] = club["players"][:4]
        cup = definition_from_dict(definition(start_date="2026-01-05"))
        state = replace(self.state(), tournament_definitions=(cup,))
        self.assertIs(state.with_tournament_entry("cup", state.selected_team_id), state)
        self.assertIsNone(state.tournament("cup"))

    def test_screen_shows_actual_field_and_waits_when_no_substitute_can_play(self):
        from run_realtime_season import RealtimeSeasonApp
        self.clubs[0]["players"] = list(RIVALS[0][:4])
        cup = definition_from_dict(definition(start_date="2026-01-05"))
        state = replace(self.state(), tournament_definitions=(cup,))
        state = state.with_tournament_entry("cup", state.selected_team_id)
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = RealtimeSeasonApp(root, self.store, state)
        app.show_screen("competitions")
        app.competition_list.selection_set("cup")
        app.preview_competition()
        row = app.competition_list.item("cup", "values")
        self.assertEqual(str(row[2]), "3")
        self.assertIn("2026-01-08", row[1])
        self.assertIn("全4試合", app.competition_info.get())
        for club in self.clubs:
            club["players"] = club["players"][:4]
        app.state = replace(self.state(), tournament_definitions=(cup,))
        app.refresh()
        app.competition_list.selection_set("cup")
        app.preview_competition()
        app.competition_enter_button.invoke()
        self.assertIsNone(app.state.tournament("cup"))
        self.assertIn("待っています", app.status.get())

    def test_version_22_loads_without_new_allocation_metadata(self):
        state = self.state()
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 22
        for club in data["opponent_teams"]:
            club.pop("initial_shared_members")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.store.load_or_create(), state)


if __name__ == "__main__":
    unittest.main()
