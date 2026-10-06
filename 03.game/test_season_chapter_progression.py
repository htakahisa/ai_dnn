"""Championship gates and shared player assets across league contexts."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_config as config
import realtime_season_leagues as leagues
import realtime_season_teams as teams
from realtime_season import SeasonSaveError, SeasonStore, new_season
from season_competitions import SeriesScore, next_match
from season_league_ui import SeasonChapterSelection
from season_transfers import TransferOffer


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
EARLY = ("Aspas", "valyn", "trent", "leaf", "tex")
MIDDLE = ("Boostio", "Ethan", "jawgemo", "C0M", "Demon1")


class ChapterProgressionTests(unittest.TestCase):
    def setUp(self):
        catalog = {name: replace(p, monthly_salary=100_000, debut_chapter=1)
                   for name, p in character_stats.CHARACTER_TABLE.items()}
        catalog["Boostio"] = replace(catalog["Boostio"], debut_chapter=2)
        self.team_definitions = [dict(name="Early", players=list(EARLY), debut_chapter=1),
                                 dict(name="Middle", players=list(MIDDLE), debut_chapter=2),
                                 dict(name="Late", players=[], debut_chapter=3)]
        self.cup = dict(id="furina", name="フリーナ杯", start_date="01-02", visible_from="01-01",
                        team_count=2, format="single_elimination", prizes={1: 9_000_000},
                        normal_maps_to_win=1, grand_final_maps_to_win=1)
        for context in (patch.dict(character_stats.CHARACTER_TABLE, catalog),
                        patch.object(config, "INITIAL_OWNED_PLAYERS", OWN),
                        patch.object(teams, "SEASON_TEAMS", self.team_definitions),
                        patch.object(calendar, "TOURNAMENTS", [self.cup]),
                        patch.object(calendar, "START_DATE", "2026-01-01"),
                        patch.object(leagues, "LEAGUE_NAMES", {1: "地域", 2: "国内", 3: "世界"})):
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "career.json")

    def play_final(self, state, *, win=True, event_id="furina_2026"):
        if not state.teams:
            state = state.with_roster(OWN).with_confirmed_team()
        state = state.with_tournament_entry(event_id, state.teams[0].id)
        if state.date.isoformat() == "2026-01-01":
            state = state.advance_days()
        match, _ = next_match(state.tournament_definition(event_id), state.tournament(event_id))
        own_left = match.left == state.club_id
        return state.with_tournament_result(event_id, SeriesScore(
            match.id, match.left, match.right,
            match.maps_to_win if own_left == win else 0,
            match.maps_to_win if own_left != win else 0))

    def test_furina_championship_unlocks_one_chapter_and_persists(self):
        state = self.play_final(new_season(OWN))
        self.assertEqual(state.tournament("furina_2026").ranking[0], state.club_id)
        self.assertEqual(state.unlocked_chapters, (1, 2))
        self.assertEqual(state.money, 10_000_000)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        selector = SeasonChapterSelection(root, lambda chapter: None, state=state)
        self.assertEqual(set(selector.buttons), {1, 2})
        self.assertNotIn(3, selector.buttons)

    def test_loss_and_other_cup_championship_do_not_unlock(self):
        lost = self.play_final(new_season(OWN), win=False)
        self.assertEqual(lost.unlocked_chapters, (1,))
        with self.assertRaisesRegex(SeasonSaveError, "フリーナ杯"):
            lost.with_chapter(2)
        with patch.object(calendar, "TOURNAMENTS", [{**self.cup, "id": "other", "name": "別大会"}]):
            other = self.play_final(new_season(OWN), event_id="other_2026")
        self.assertEqual(other.unlocked_chapters, (1,))

    def test_nonparticipating_npc_champion_does_not_unlock(self):
        another = tuple(n for n in character_stats.CHARACTER_TABLE if n not in (*OWN, *EARLY, *MIDDLE))[:5]
        self.team_definitions.append(dict(name="Extra", players=list(another), debut_chapter=1))
        state = new_season(OWN).advance_days(2)
        self.assertTrue(state.tournament("furina_2026").completed)
        self.assertIsNone(state.tournament("furina_2026").own_team_id)
        self.assertEqual(state.unlocked_chapters, (1,))

    def test_second_chapter_requires_its_own_championship_to_unlock_third(self):
        first = self.play_final(new_season(OWN))
        second = first.with_chapter(2)
        self.assertEqual(second.unlocked_chapters, (1, 2))
        with self.assertRaises(SeasonSaveError):
            second.with_chapter(3)
        second = self.play_final(second)
        self.assertEqual(second.unlocked_chapters, (1, 2, 3))
        self.assertEqual(second.money, first.money + 9_000_000)
        self.assertEqual(second.with_chapter(3).chapter, 3)

    def test_returning_to_won_chapter_cannot_replay_prizes_or_unlock_third(self):
        first = self.play_final(new_season(OWN))
        second = first.with_chapter(2)
        back = second.with_chapter(1)
        self.assertEqual(back.money, first.money)
        self.assertEqual(back.tournaments, first.tournaments)
        self.assertEqual(back.rated_results, first.rated_results)
        self.assertEqual(back.unlocked_chapters, (1, 2))
        self.assertIsNone(next_match(back.tournament_definition("furina_2026"), back.tournament("furina_2026"))[0])
        self.store.save(back)
        self.assertEqual(self.store.load_or_create().unlocked_chapters, (1, 2))

    def test_player_growth_loyalties_contracts_rating_and_spent_money_survive_roundtrip(self):
        first = self.play_final(new_season(OWN))
        first = replace(first, money=50_000_000,
                        owned_players=tuple(replace(p, iq=p.iq + 7, hit_pct=p.hit_pct + .1,
                                                    research_level=3, aim_lab_level=4)
                                            for p in first.owned_players))
        first = replace(first, contracts=tuple(replace(c, kind="year3", duration_months=36)
                                               if c.player_name == "Leo" else c for c in first.contracts))
        second = first.with_chapter(2)
        self.assertEqual(second.contracts, first.contracts)
        self.assertEqual(second.owned_players, first.owned_players)
        self.assertEqual(second.rating(second.club_id), first.rating(first.club_id))
        self.assertTrue(first.pair_days.items() <= second.pair_days.items())
        second = second.with_trained_player("Leo", "research")
        second = second.with_scouted_player("Boostio", "year1")
        self.assertIsNotNone(second.player("Boostio"))
        paid_money = second.money
        self.store.save(second)
        second = self.store.load_or_create()
        back = second.with_chapter(1)
        self.assertEqual(back.money, paid_money)
        self.assertEqual(back.date, second.date)
        self.assertEqual(back.owned_players, second.owned_players)
        self.assertEqual(back.contracts, second.contracts)
        self.assertEqual(back.rating(back.club_id), second.rating(second.club_id))
        self.assertEqual(back.pair_days, second.pair_days)
        self.assertEqual(back.team_loyalty("Leo"), second.team_loyalty("Leo"))
        self.assertEqual(back.player("Leo").research_level, 4)
        self.assertIsNotNone(back.player("Boostio"))
        self.assertNotIn("Boostio", [p.name for p in back.scout_players])
        self.store.save(back)
        self.assertEqual(self.store.load_or_create(), back)

    def test_initial_distribution_is_only_once_and_first_chapter_only(self):
        state = self.store.load_or_create()
        candidates = state.starter_candidates
        self.assertTrue(state.starter_selection_pending)
        self.assertEqual(self.store.load_or_create().starter_candidates, candidates)
        with self.assertRaises(SeasonSaveError):
            new_season(chapter=2)
        with self.assertRaises(SeasonSaveError):
            replace(state, unlocked_chapters=(1, 2)).with_chapter(2)
        state = state.with_initial_selection(OWN)
        state = self.play_final(state)
        second = state.with_chapter(2)
        self.assertFalse(second.starter_selection_pending)
        self.assertEqual(len(second.owned_players), 5)
        self.assertEqual(second.starter_candidates, candidates)
        with self.assertRaises(SeasonSaveError):
            second.with_initial_selection(OWN)
        self.store.save(second)
        self.assertFalse(self.store.load_or_create().starter_selection_pending)

    def test_rival_roster_money_contracts_and_pending_offer_survive_chapter_return(self):
        first = self.play_final(new_season(OWN))
        club = first.opponent_teams[0]
        club = replace(club, money=17_654_321)
        offer = TransferOffer("persistent-offer", club.id, "Leo", first.game_month, 3_000_000,
                              contract_kind="year1", contract_months=12, monthly_salary=100_000)
        first = replace(first, opponent_teams=(club,), transfer_offers=(offer,))
        second = first.with_chapter(2)
        self.store.save(second)
        second = self.store.load_or_create()
        back = second.with_chapter(1)
        self.assertEqual(back.opponent_teams, (club,))
        self.assertEqual(back.transfer_offers, (offer,))
        self.assertEqual(back.money, first.money)
        back = back.with_chapter(2)
        for name, value in first.team_loyalties.items():
            self.assertEqual(back.team_loyalties[name][first.club_id], value[first.club_id])

    def test_old_save_migrates_completed_championship_without_rewriting_original(self):
        first = self.play_final(new_season(OWN))
        self.store.save(first)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 30
        data.pop("unlocked_chapters")
        data.pop("chapter_environments")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.unlocked_chapters, (1, 2))
        self.assertEqual(loaded.contracts, first.contracts)
        self.assertEqual(loaded.money, first.money)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.store.save(loaded.with_chapter(2))
        self.assertEqual(self.store.load_or_create().chapter, 2)

    def test_cannot_switch_with_unfinished_event_or_paid_day_pending(self):
        first = replace(new_season(OWN).with_roster(OWN).with_confirmed_team(), unlocked_chapters=(1, 2))
        registered = first.with_tournament_entry("furina_2026", first.teams[0].id)
        with self.assertRaisesRegex(SeasonSaveError, "大会終了"):
            registered.with_chapter(2)
        self.assertEqual(registered.with_chapter(1), registered)
        with self.assertRaisesRegex(SeasonSaveError, "日付進行"):
            replace(first, day_advance_pending=True).with_chapter(2)

    def test_arriving_after_annual_start_does_not_retroactively_enter_that_year(self):
        first = self.play_final(new_season(OWN)).advance_days(3)
        second = first.with_chapter(2)
        self.assertIsNone(second.tournament_definition("furina_2026"))
        self.assertIsNotNone(second.tournament_definition("furina_2027"))
        self.assertEqual(second.date, first.date)
        self.assertEqual(second.money, first.money)
        self.assertFalse(second.tournaments)
        advanced = second.advance_days(4)
        self.assertIsNone(advanced.tournament_definition("furina_2026"))
        self.assertFalse(advanced.tournaments)
        self.assertEqual(advanced.rated_results, ())
        self.assertEqual(advanced.chapter_started_on, second.date.isoformat())

    def test_october_arrival_omits_all_past_cups_until_the_next_year(self):
        first = self.play_final(new_season(OWN))
        first = replace(first, game_date="2026-10-20", game_month=9, monthly_events_through=9)
        first = replace(first, contracts=tuple(replace(c, kind="year3", duration_months=36)
                                               for c in first.contracts))
        cups = [{**self.cup, "id": name, "start_date": day}
                for name, day in (("kachina", "01-15"), ("lisa", "03-10"),
                                  ("jean", "04-20"), ("lohen", "06-30"), ("furina", "09-30"))]
        other = tuple(n for n in character_stats.CHARACTER_TABLE if n not in (*OWN, *EARLY, *MIDDLE))[:5]
        self.team_definitions.append(dict(name="Extra2", players=list(other), debut_chapter=2))
        with patch.object(calendar, "TOURNAMENTS", cups):
            second = first.with_chapter(2)
            self.assertEqual(second.chapter_started_on, "2026-10-20")
            self.assertTrue(all(e.start_date.startswith("2027-") for e in second.tournament_definitions))
            before = [(c.id, c.money) for c in second.opponent_teams]
            advanced = second.advance_days(5)
            self.assertFalse(advanced.tournaments)
            self.assertEqual(advanced.rated_results, ())
            self.assertEqual([(c.id, c.money) for c in advanced.opponent_teams], before)
            self.assertFalse(any(e["種別"] == "大会自動開催（自チーム不参加）" for e in advanced.history))
            self.store.save(advanced)
            loaded = self.store.load_or_create()
            self.assertEqual(loaded, advanced)
            new_year = loaded.advance_days(73)
            self.assertEqual(new_year.date.isoformat(), "2027-01-06")
            self.assertFalse(new_year.tournaments)
            self.assertIsNotNone(new_year.tournament_definition("kachina_2027"))
            registered = new_year.with_tournament_entry("kachina_2027", new_year.teams[0].id)
            self.assertEqual(registered.tournament("kachina_2027").own_team_id, registered.club_id)

    def test_events_on_and_after_chapter_arrival_are_still_available(self):
        first = self.play_final(new_season(OWN)).advance_days(3)
        for day in ("01-05", "01-10"):
            with self.subTest(day=day), patch.object(calendar, "TOURNAMENTS", [{**self.cup, "start_date": day}]):
                second = first.with_chapter(2)
                self.assertIsNotNone(second.tournament_definition("furina_2026"))
                registered = second.with_tournament_entry("furina_2026", second.teams[0].id)
                self.assertEqual(registered.tournament("furina_2026").own_team_id, second.club_id)

    def test_chapter_return_keeps_original_arrival_date(self):
        first = self.play_final(new_season(OWN)).advance_days(3)
        second = first.with_chapter(2).advance_days(2)
        back = second.with_chapter(1).advance_days(2).with_chapter(2)
        self.assertEqual(back.chapter_started_on, first.date.isoformat())
        self.assertIsNone(back.advance_days().tournament_definition("furina_2026"))

    def test_legacy_arrival_date_is_recovered_without_changing_save(self):
        first = self.play_final(new_season(OWN)).advance_days(3)
        second = first.with_chapter(2).advance_days(2)
        self.store.save(second)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 32
        data.pop("chapter_started_on")
        for environment in data["chapter_environments"]:
            environment.pop("chapter_started_on")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        original = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.chapter_started_on, first.date.isoformat())
        self.assertEqual(self.store.path.read_bytes(), original)
        self.assertIsNone(loaded.advance_days().tournament_definition("furina_2026"))

    def test_legacy_prearrival_npc_tournament_is_removed(self):
        from season_competitions import TournamentProgress

        first = self.play_final(new_season(OWN)).advance_days(3)
        second = first.with_chapter(2)
        past = first.tournament_definition("furina_2026")
        second = replace(second, tournament_definitions=(*second.tournament_definitions, past),
                         tournaments=(TournamentProgress(past.id, None, completed=True, declined=True),))
        self.store.save(second)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 32
        data.pop("chapter_started_on")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        loaded = self.store.load_or_create()
        self.assertIsNone(loaded.tournament_definition(past.id))
        self.assertIsNone(loaded.tournament(past.id))
        self.assertEqual(loaded.money, second.money)

    def test_invalid_chapter_start_is_rejected(self):
        second = self.play_final(new_season(OWN)).with_chapter(2)
        for value in ("2025-12-31", "2027-01-01", 2, "invalid"):
            with self.subTest(value=value), self.assertRaises(SeasonSaveError):
                replace(second, chapter_started_on=value).validate()

    def test_corrupt_archived_club_is_rejected_without_overwriting_save(self):
        state = self.play_final(new_season(OWN)).with_chapter(2)
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["chapter_environments"][0]["opponent_teams"][0]["money"] = "invalid"
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        with self.assertRaises(SeasonSaveError):
            self.store.load_or_create()
        self.assertEqual(self.store.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
