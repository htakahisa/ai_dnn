from dataclasses import replace
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from character_stats import get_by_name
import realtime_season_config
import realtime_season_teams
from realtime_season import SAVE_VERSION, SeasonSaveError, SeasonStore, SeasonTeam, new_season
from run_realtime_season import RealtimeSeasonApp


PLAYERS = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer", "Meiy")
SECOND_PLAYERS = ("Demon1", "Ethan", "jawgemo", "Boostio", "C0M")
RIVAL_PLAYERS = ("Aspas", "valyn", "trent", "leaf", "tex", "Sato")


class SeasonPersistenceTest(unittest.TestCase):
    def setUp(self):
        config_patch = patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", [])
        config_patch.start()
        self.addCleanup(config_patch.stop)
        teams_patch = patch.object(realtime_season_teams, "SEASON_TEAMS", [])
        teams_patch.start()
        self.addCleanup(teams_patch.stop)
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "season" / "save.json"
        self.store = SeasonStore(self.path)

    def test_configured_initial_players_are_saved_on_first_launch(self):
        with patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", [*PLAYERS, "まーやまくん"]):
            state = self.store.load_or_create()
        self.assertEqual(tuple(p.name for p in state.starter_candidates), (*PLAYERS, "まーやまくん"))
        self.assertEqual(state.owned_players, ())
        self.assertTrue(state.starter_selection_pending)
        self.assertEqual(state.roster, ())
        self.assertEqual(SeasonStore(self.path).load_or_create(), state)

    def test_config_changes_do_not_overwrite_existing_inventory_abilities_or_roster(self):
        with patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", list(PLAYERS)):
            state = self.store.load_or_create().with_initial_selection(PLAYERS[:5]).with_roster(PLAYERS[:5])
        state = replace(state, owned_players=(replace(state.owned_players[0], iq=180), *state.owned_players[1:]))
        self.store.save(state)
        original = self.path.read_bytes()
        with patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", ["NoSuchPlayer"]):
            self.assertEqual(SeasonStore(self.path).load_or_create(), state)
        self.assertEqual(self.path.read_bytes(), original)

    def test_season_clubs_and_reserves_are_saved_independently_of_other_modes(self):
        config = [{"name": "ライバル", "players": list(RIVAL_PLAYERS)}]
        with patch.object(realtime_season_teams, "SEASON_TEAMS", config):
            state = new_season(()).with_added_players(PLAYERS)
        club = state.opponent_teams[0]
        self.assertEqual(club.members, RIVAL_PLAYERS)
        self.assertEqual(club.roster, RIVAL_PLAYERS[:5])
        self.assertIsNone(state.player("Aspas"))
        self.store.save(state)
        with patch("character_stats.CHARACTER_TABLE", {}):
            loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertEqual(loaded.opponent_teams[0].players[5].name, "Sato")

    def test_club_match_settings_import_and_survive_restart(self):
        state = new_season(PLAYERS)
        config = [{"name": "ライバル", "players": list(RIVAL_PLAYERS), "igl": "Aspas", "carrier": "trent", "ai": "fnatic_v3"}]
        with patch.object(realtime_season_teams, "SEASON_TEAMS", config):
            state = self.store.import_season_teams(state)
        club = self.store.load_or_create().opponent_teams[0]
        self.assertEqual((club.igl, club.carrier, club.ai), ("Aspas", "trent", "fnatic_v3"))
        self.assertEqual((club.effective_igl, club.effective_carrier), ("Aspas", "trent"))
        self.assertEqual(self.store.load_or_create(), state)

    def test_invalid_club_igl_carrier_or_ai_cannot_overwrite_save(self):
        state = new_season(PLAYERS)
        self.store.save(state)
        original = self.path.read_bytes()
        for overrides in ({"igl": "Sato"}, {"carrier": "Sato"}, {"igl": "unknown"}, {"carrier": ""}, {"ai": "missing"}, {"ai": None}, {"ai": []}):
            config = [{"name": "ライバル", "players": list(RIVAL_PLAYERS), **overrides}]
            with self.subTest(overrides=overrides), patch.object(realtime_season_teams, "SEASON_TEAMS", config):
                with self.assertRaises(SeasonSaveError):
                    self.store.import_season_teams(state)
                self.assertEqual(self.path.read_bytes(), original)

    def test_old_club_save_without_match_settings_has_compatible_defaults(self):
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "ライバル", "players": list(RIVAL_PLAYERS)}]):
            state = new_season(())
            self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for item in data["opponent_teams"]:
            for key in ("igl", "carrier", "ai"):
                item.pop(key)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        club = loaded.opponent_teams[0]
        self.assertEqual(club.effective_igl, max(club.players[:5], key=lambda p: p.iq).name)
        self.assertEqual(club.effective_carrier, "Aspas")
        self.assertEqual(club.ai, "toru_ai_v3.1")

    def test_invalid_world_config_and_ownership_overlap_never_create_save(self):
        club = {"name": "ライバルA", "players": list(RIVAL_PLAYERS)}
        bad_configs = (
            "invalid",
            [{"name": "ライバル", "players": "Aspas"}],
            [{"name": "ライバル", "players": [*RIVAL_PLAYERS[:5], "Aspas"]}],
            [{"name": "ライバル", "players": [*RIVAL_PLAYERS[:4], "NoSuchPlayer"]}],
            [club, dict(club, players=list(SECOND_PLAYERS))],
        )
        for config in bad_configs:
            with self.subTest(config=config), patch.object(realtime_season_teams, "SEASON_TEAMS", config):
                with self.assertRaises(SeasonSaveError):
                    self.store.load_or_create()
                self.assertFalse(self.path.exists())
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [club]), patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", [*PLAYERS[:4], "Sato"]):
            state = self.store.load_or_create().with_initial_selection([*PLAYERS[:4], "Sato"])
            self.assertIsNotNone(state.player("Sato"))
            self.assertIsNone(state.opponent_owner("Sato"))

    def test_import_updates_world_but_keeps_saved_abilities_and_user_state(self):
        state = new_season(PLAYERS).with_roster(PLAYERS[:5]).with_confirmed_team()
        state = state.with_selected_team(state.teams[0].id)
        self.store.save(state)
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "ライバル", "players": list(RIVAL_PLAYERS)}]):
            imported = self.store.import_season_teams(state)
        club = imported.opponent_teams[0]
        club = replace(club, players=(replace(club.players[0], iq=180), *club.players[1:]))
        imported = imported.with_opponent_teams((club,))
        self.store.save(imported)
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "新所属", "players": list(RIVAL_PLAYERS)}]):
            updated = self.store.import_season_teams(imported)
        self.assertEqual(updated.owned_players, state.owned_players)
        self.assertEqual(updated.teams, state.teams)
        self.assertEqual(updated.selected_team, state.selected_team)
        self.assertEqual(updated.roster, state.roster)
        self.assertEqual(updated.opponent_teams[0].name, "新所属")
        self.assertEqual(updated.opponent_teams[0].players[0].iq, 180)
        self.assertEqual(self.store.load_or_create(), updated)

    def test_import_prioritizes_owned_players_and_adding_rivals_is_still_rejected(self):
        state = new_season(PLAYERS)
        self.store.save(state)
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "ライバル", "players": list(PLAYERS[:5])}]):
            imported = self.store.import_season_teams(state)
        self.assertEqual(imported.owned_players, state.owned_players)
        self.assertFalse(imported.opponent_teams[0].players)
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "ライバル", "players": list(RIVAL_PLAYERS)}]):
            state = self.store.import_season_teams(imported)
        for name in ("Aspas", "Sato"):
            with self.subTest(name=name), self.assertRaisesRegex(SeasonSaveError, f"{name}.*ライバル"):
                state.with_added_players((name,))

    def test_presets_reuse_players_in_drafts_save_and_old_data(self):
        state = new_season(PLAYERS).with_roster(PLAYERS[:5]).with_confirmed_team()
        team_a = state.teams[0]
        self.assertEqual(state.with_new_team().with_roster(("Leo",)).roster, ("Leo",))
        self.store.save(state)
        original = self.path.read_bytes()
        duplicate = SeasonTeam("duplicate-team", "重複チーム", PLAYERS[1:])
        self.store.save(replace(state, teams=(team_a, duplicate)))
        self.assertEqual(len(self.store.load_or_create().teams), 2)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 2
        data.pop("opponent_teams")
        self.path.write_text(json.dumps(data), encoding="utf-8")
        invalid = self.path.read_bytes()
        self.assertEqual(len(self.store.load_or_create().teams), 2)
        self.assertEqual(self.path.read_bytes(), invalid)

    def test_valid_version_two_save_keeps_user_team_and_upgrades_when_imported(self):
        state = new_season(PLAYERS).with_roster(PLAYERS[:5]).with_confirmed_team()
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 2
        data.pop("opponent_teams")
        for player in data["owned_players"]:
            player["loyalty"] *= 10
        self.path.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "ライバル", "players": list(RIVAL_PLAYERS)}]):
            loaded = self.store.load_or_create()
            self.assertEqual(loaded, replace(state, contracts=tuple(replace(c, signed_on=None) for c in state.contracts)))
            imported = self.store.import_season_teams(loaded)
        self.assertEqual(imported.teams, state.teams)
        self.assertEqual(len(imported.opponent_teams), 1)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["version"], SAVE_VERSION)

    def test_invalid_initial_config_does_not_create_save(self):
        for config in (["Leo", "NoSuchPlayer"], ["Leo", "Leo"], "Leo", [None], [""]):
            with self.subTest(config=config), patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", config):
                with self.assertRaises(SeasonSaveError):
                    self.store.load_or_create()
                self.assertFalse(self.path.exists())

    def test_legacy_empty_save_and_manual_inventory_survive_restart(self):
        self.store.save(new_season(()))
        state = self.store.load_or_create()
        self.assertEqual(state.owned_players, ())
        self.assertEqual(state.roster, ())
        self.assertTrue(self.path.is_file())
        state = state.with_added_players((*PLAYERS, "まーやまくん"))
        state = state.with_team_name("育成チーム")
        self.store.save(state)
        self.assertEqual(SeasonStore(self.path).load_or_create(), state)

    def test_multiple_teams_and_selected_team_survive_restart(self):
        state = new_season((*PLAYERS, *SECOND_PLAYERS)).with_preset_name("チームA").with_roster(PLAYERS[:5]).with_confirmed_team()
        team_a = state.teams[0]
        state = state.with_new_team().with_preset_name("チームB").with_roster(SECOND_PLAYERS).with_confirmed_team()
        state = state.with_selected_team(team_a.id)
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual([team.name for team in loaded.teams], ["チームA", "チームB"])
        self.assertEqual(loaded.selected_team, team_a)
        self.assertEqual(loaded.roster, SECOND_PLAYERS)

    def test_editing_registered_team_keeps_id_and_updates_only_on_confirmation(self):
        state = new_season(PLAYERS).with_roster(PLAYERS[:5]).with_confirmed_team()
        original = state.teams[0]
        state = state.with_selected_team(original.id).with_roster(PLAYERS[1:]).with_preset_name("変更したチーム")
        self.assertEqual(state.selected_team, original)
        state = state.with_confirmed_team()
        self.assertEqual(len(state.teams), 1)
        self.assertEqual(state.selected_team.id, original.id)
        self.assertEqual(state.selected_team.roster, PLAYERS[1:])
        self.assertEqual(state.selected_team.name, "変更したチーム")

    def test_incomplete_duplicate_teams_and_unknown_selection_are_rejected(self):
        state = new_season(PLAYERS).with_roster(PLAYERS[:3])
        with self.assertRaises(SeasonSaveError):
            state.with_confirmed_team()
        state = state.with_roster(PLAYERS[:5]).with_confirmed_team()
        with self.assertRaises(SeasonSaveError):
            state.with_new_team().with_roster(PLAYERS[1:]).with_preset_name(state.preset_name).with_confirmed_team()
        with self.assertRaises(SeasonSaveError):
            state.with_selected_team("unknown-id")
        with self.assertRaises(SeasonSaveError):
            state.with_roster(()).without_player("Leo")

    def test_version_one_save_migrates_complete_team_and_preserves_draft(self):
        for roster in (PLAYERS[:5], PLAYERS[:3]):
            with self.subTest(roster=roster):
                state = new_season(PLAYERS).with_roster(roster).with_team_name("旧チーム")
                self.store.save(state)
                data = json.loads(self.path.read_text(encoding="utf-8"))
                data["version"] = 1
                for player in data["owned_players"]:
                    player["loyalty"] *= 10
                for key in ("teams", "editing_team_id", "selected_team_id"):
                    data.pop(key)
                self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                original = self.path.read_bytes()
                loaded = self.store.load_or_create()
                self.assertEqual(loaded.owned_players, state.owned_players)
                self.assertEqual(loaded.roster, roster)
                self.assertEqual(len(loaded.teams), 1 if len(roster) == 5 else 0)
                if loaded.teams:
                    self.assertEqual(loaded.teams[0].name, "旧チーム")
                    self.assertEqual(loaded.teams[0].roster, roster)
                self.assertEqual(self.path.read_bytes(), original)
                self.store.save(loaded)
                self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["version"], SAVE_VERSION)
                self.assertEqual(self.store.load_or_create(), loaded)

    def test_draft_complete_and_replaced_rosters_survive_restart(self):
        state = new_season(PLAYERS).with_roster(PLAYERS[:3])
        self.store.save(state)
        self.assertEqual(self.store.load_or_create().roster, PLAYERS[:3])
        state = state.with_roster(PLAYERS[:5])
        self.assertTrue(state.roster_ready)
        self.store.save(state)
        state = self.store.load_or_create().with_roster((*PLAYERS[:4], "Meiy"))
        self.store.save(state)
        self.assertEqual(SeasonStore(self.path).load_or_create(), state)

    def test_invalid_lineup_cannot_overwrite_save(self):
        state = new_season(PLAYERS).with_roster(PLAYERS[:5])
        self.store.save(state)
        original = self.path.read_bytes()
        for roster in (PLAYERS, ("Leo", "Leo"), ("Aspas",), ({"name": "Leo"},)):
            with self.subTest(roster=roster), self.assertRaises(SeasonSaveError):
                self.store.save(replace(state, roster=roster))
            self.assertEqual(self.path.read_bytes(), original)

    def test_manual_names_validate_as_a_batch(self):
        state = new_season(())
        for names in (("Leo", "unknown"), ("Leo", "Leo")):
            with self.subTest(names=names), self.assertRaises(SeasonSaveError):
                state.with_added_players(names)
        self.assertEqual(state.owned_players, ())
        state = state.with_added_players(("Leo",))
        with self.assertRaises(SeasonSaveError):
            state.with_added_players(("Leo",))

    def test_owned_player_removal_requires_removing_from_roster_first(self):
        state = new_season(PLAYERS).with_roster(("Leo",))
        with self.assertRaises(SeasonSaveError):
            state.without_player("Leo")
        state = state.with_roster(()).without_player("Leo")
        self.store.save(state)
        self.assertIsNone(self.store.load_or_create().player("Leo"))

    def test_saved_abilities_are_individual_snapshots(self):
        state = new_season(PLAYERS)
        grown = replace(state.owned_players[0], iq=180, reaction=160)
        state = replace(state, owned_players=(grown, *state.owned_players[1:]))
        self.store.save(state)
        with patch("character_stats.CHARACTER_TABLE", {}):
            loaded = self.store.load_or_create()
        self.assertEqual(loaded.player("Leo"), grown)
        self.assertNotEqual(get_by_name("Leo").iq, loaded.player("Leo").iq)

    def test_corrupt_or_unsupported_save_is_preserved(self):
        self.store.save(new_season(()))
        valid = json.loads(self.path.read_text(encoding="utf-8"))
        bad_version = dict(valid, version=999)
        bad_player = dict(valid, owned_players=[dict(name="Leo")])
        for payload in (b"{broken", b"\xff", b"null", json.dumps(bad_version).encode(), json.dumps(bad_player).encode()):
            with self.subTest(payload=payload):
                self.path.write_bytes(payload)
                with self.assertRaises(SeasonSaveError):
                    self.store.load_or_create()
                self.assertEqual(self.path.read_bytes(), payload)

    def test_failed_atomic_replace_keeps_previous_save_and_cleans_temporary_file(self):
        state = new_season(())
        self.store.save(state)
        original = self.path.read_bytes()
        with patch("realtime_season.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.store.save(state.with_added_players(PLAYERS))
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(set(self.path.parent.iterdir()), {self.path, self.store.history_path})


class SeasonScreenTest(unittest.TestCase):
    def setUp(self):
        config_patch = patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", [])
        config_patch.start()
        self.addCleanup(config_patch.stop)
        teams_patch = patch.object(realtime_season_teams, "SEASON_TEAMS", [])
        teams_patch.start()
        self.addCleanup(teams_patch.stop)
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.store = SeasonStore(Path(self.directory.name) / "save.json")
        self.store.save(new_season(()))
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.destroy_root)
        self.app = RealtimeSeasonApp(self.root, self.store, self.store.load_or_create())

    def destroy_root(self):
        if self.root.winfo_exists():
            self.root.destroy()

    def select_inventory(self, name):
        index = next(i for i, player in enumerate(self.app.state.owned_players) if player.name == name)
        self.app.players.selection_set(str(index))
        self.app.select_player()

    def register_team(self, name, players):
        self.app.show_editor()
        self.app.preset_name.set(name)
        self.app.save_preset_name()
        for player in players:
            self.select_inventory(player)
            self.app.add_player()
        self.app.confirm()

    def test_home_editor_preparation_multiple_team_selection_and_restart(self):
        self.assertEqual(self.app.current_screen, "home")
        self.app.show_editor()
        self.assertEqual(self.app.current_screen, "editor")
        self.app.name_input.insert("1.0", ",".join((*PLAYERS, *SECOND_PLAYERS)))
        self.app.register_players()
        self.register_team("チームA", PLAYERS[:5])
        self.app.new_team()
        self.register_team("チームB", SECOND_PLAYERS)
        self.app.show_home()
        self.assertEqual(len(self.app.home_teams.get_children()), 1)
        self.app.show_preparation()
        self.assertEqual(self.app.current_screen, "preparation")
        self.assertEqual(tuple(self.app.prep_team_menu["values"]), ("チームA", "チームB"))
        self.app.prep_team_choice.set("チームA")
        self.app.preview_preparation()
        displayed = tuple(self.app.prep_roster.item(item, "values")[1] for item in self.app.prep_roster.get_children())
        self.assertEqual(displayed, PLAYERS[:5])
        self.app.confirm_preparation()
        self.assertEqual(self.app.state.selected_team.name, "チームA")
        self.assertIn("チームA", self.app.home_selected.get())
        self.app.show_editor()
        self.app.edit_team_choice.set("チームA")
        self.app.edit_saved_team()
        self.assertEqual(self.app.state.roster, PLAYERS[:5])
        self.assertEqual(self.app.preset_name.get(), "チームA")
        self.app.show_preparation()
        self.app.prep_team_choice.set("チームB")
        self.app.confirm_preparation()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.selected_team.name, "チームB")
        self.root.destroy()
        self.root = tk.Tk()
        self.root.withdraw()
        restored = RealtimeSeasonApp(self.root, self.store, loaded)
        self.assertEqual(restored.current_screen, "home")
        restored.show_preparation()
        self.assertEqual(restored.prep_team_choice.get(), "チームB")
        self.assertEqual(len(restored.prep_roster.get_children()), 5)

    def test_preparation_guides_user_until_team_is_confirmed(self):
        self.app.show_preparation()
        self.assertEqual(str(self.app.prep_confirm_button["state"]), "disabled")
        self.assertIn("編成プリセットがありません", self.app.prep_hint.get())
        self.app.show_editor()
        self.app.name_input.insert("1.0", ",".join(PLAYERS))
        self.app.register_players()
        for name in PLAYERS[:5]:
            self.select_inventory(name)
            self.app.add_player()
        self.app.show_preparation()
        self.assertEqual(tuple(self.app.prep_team_menu["values"]), ())
        self.app.show_editor()
        self.app.confirm()
        self.app.show_preparation()
        self.assertEqual(len(self.app.prep_team_menu["values"]), 1)

    def test_rival_clubs_are_visible_and_cannot_be_selected_or_owned(self):
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "ライバル", "players": list(RIVAL_PLAYERS)}]):
            imported = self.store.import_season_teams(self.app.state)
        self.app.commit(imported, "所属設定を読み込みました。")
        club = imported.opponent_teams[0]
        self.assertEqual(self.app.home_teams.item(club.id, "values")[0], "他チーム")
        self.assertIn("Sato", self.app.home_teams.set(club.id, "players"))
        self.app.show_preparation()
        self.assertEqual(tuple(self.app.prep_team_menu["values"]), ())
        self.app.prep_team_choice.set("ライバル")
        self.app.confirm_preparation()
        self.assertIsNone(self.app.state.selected_team)
        self.app.show_editor()
        self.app.name_input.insert("1.0", "Leo, Sato")
        self.app.register_players()
        self.assertEqual(self.app.state.owned_players, ())
        self.assertIn("ライバル", self.app.status.get())

    def test_player_in_another_preset_can_be_added_to_new_lineup(self):
        self.app.name_input.insert("1.0", ",".join(PLAYERS))
        self.app.register_players()
        self.register_team("チームA", PLAYERS[:5])
        self.app.new_team()
        self.select_inventory("Leo")
        self.assertEqual(str(self.app.add_button["state"]), "normal")
        self.assertIn(self.app.state.team_name, self.app.details.get())
        self.app.add_player()
        self.assertEqual(self.app.state.roster, ("Leo",))
        self.select_inventory("Meiy")
        self.assertEqual(str(self.app.add_button["state"]), "normal")
        self.app.add_player()
        self.assertEqual(self.app.state.roster, ("Leo", "Meiy"))

    def prepare_scrim(self):
        self.app.name_input.insert("1.0", ",".join(PLAYERS))
        self.app.register_players()
        self.register_team("チームA", PLAYERS[:5])
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "ライバル", "players": list(RIVAL_PLAYERS)}]):
            imported = self.store.import_season_teams(self.app.state)
        self.app.commit(imported, "所属設定を読み込みました。")
        self.app.show_preparation()
        self.app.prep_team_choice.set("チームA")
        self.app.opponent_choice.set("ライバル")
        self.app.preview_preparation()

    def test_scrim_start_passes_preparation_settings_and_displays_result(self):
        self.prepare_scrim()
        self.assertEqual(len(self.app.opponent_roster.get_children()), 5)
        self.assertNotIn("Sato", self.app.role_menus[("opponent", "IGL")]["values"])
        self.app.scrim_render.set(False)
        self.app.scrim_side.set("防衛")
        self.app.scrim_tick_ms.set("7")
        self.app.own_ai_choice.set("Fnatic v3")
        self.app.own_igl.set("Leo")
        self.app.own_spike.set("Derke")
        job = Mock()
        job.poll.return_value = {"status": "completed", "own_team": "チームA", "opponent_team": "ライバル", "own_score": 13, "opponent_score": 6, "winner": "チームA"}
        with patch("run_realtime_season.ScrimJob", return_value=job) as launch:
            self.app.start_scrim()
            self.app.start_scrim()
            launch.assert_called_once()
            request = launch.call_args.args[0]
            self.assertFalse(request["render"])
            self.assertEqual(request["initial_side"], "D")
            self.assertEqual(request["tick_time_ms"], 7)
            self.assertEqual(request["own"]["ai"], "fnatic_v3")
            self.assertEqual(request["own"]["igl"], "Leo")
            self.assertEqual(request["own"]["spike_holder"], "Derke")
            self.assertEqual(str(self.app.scrim_start_button["state"]), "disabled")
            self.assertEqual(self.app.current_screen, "preparation")
            self.assertEqual(self.store.load_or_create().selected_team.name, "チームA")
            self.app.show_home()
            self.root.after_cancel(self.app._scrim_after_id)
            self.app.poll_scrim()
            self.assertIn("13 - 6", self.app.scrim_result.get())
            self.assertIsNone(self.app.scrim_job)
            self.assertEqual(str(self.app.scrim_start_button["state"]), "normal")

    def test_opponent_file_settings_apply_on_selection_and_reach_scrim_request(self):
        self.prepare_scrim()
        config = [
            {"name": "ライバル", "players": list(RIVAL_PLAYERS), "igl": "Aspas", "carrier": "trent", "ai": "fnatic_v3"},
            {"name": "別チーム", "players": list(SECOND_PLAYERS), "igl": "Demon1", "carrier": "Boostio", "ai": "frc_v1_baseline"},
        ]
        with patch.object(realtime_season_teams, "SEASON_TEAMS", config):
            self.app.commit(self.store.import_season_teams(self.app.state), "所属設定を更新しました。")
        self.assertEqual(self.app.opponent_igl.get(), "Aspas")
        self.assertEqual(self.app.opponent_spike.get(), "trent")
        self.assertEqual(self.app.opponent_ai_choice.get(), "Fnatic v3")
        self.app.opponent_igl.set("leaf")
        self.app.opponent_spike.set("valyn")
        self.app.opponent_ai_choice.set("ロジック")
        self.app.preview_preparation()
        self.assertEqual(self.app.opponent_igl.get(), "leaf")
        self.assertEqual(self.app.opponent_spike.get(), "valyn")
        self.assertEqual(self.app.opponent_ai_choice.get(), "ロジック")
        self.app.opponent_choice.set("別チーム")
        self.app.preview_preparation()
        self.assertEqual(self.app.opponent_igl.get(), "Demon1")
        self.assertEqual(self.app.opponent_spike.get(), "Boostio")
        self.assertEqual(self.app.opponent_ai_choice.get(), "FRC v1（基礎ルール）")
        self.app.prep_team_choice.set("チームA")
        job = Mock()
        job.poll.return_value = {"status": "cancelled"}
        with patch("run_realtime_season.ScrimJob", return_value=job) as launch:
            self.app.start_scrim()
            request = launch.call_args.args[0]
            self.assertEqual(request["opponent"]["igl"], "Demon1")
            self.assertEqual(request["opponent"]["spike_holder"], "Boostio")
            self.assertEqual(request["opponent"]["ai"], "frc_v1_baseline")
        self.root.after_cancel(self.app._scrim_after_id)
        self.app.poll_scrim()

    def test_invalid_scrim_settings_do_not_launch_and_cancel_stops_active_job(self):
        self.prepare_scrim()
        self.app.scrim_render.set(False)
        self.app.own_ai_choice.set("ユーザー操作")
        self.app.update_scrim_start_state()
        self.assertEqual(str(self.app.scrim_start_button["state"]), "disabled")
        with patch("run_realtime_season.ScrimJob") as launch:
            self.app.start_scrim()
            launch.assert_not_called()
        self.app.own_ai_choice.set("ロジック")
        self.app.scrim_tick_ms.set("0")
        with patch("run_realtime_season.ScrimJob") as launch:
            self.app.start_scrim()
            launch.assert_not_called()
        self.app.scrim_tick_ms.set("100")
        job = Mock()
        job.poll.return_value = {"status": "cancelled"}
        with patch("run_realtime_season.ScrimJob", return_value=job):
            self.app.start_scrim()
        self.app.cancel_scrim()
        job.cancel.assert_called_once()
        self.root.after_cancel(self.app._scrim_after_id)
        self.app.poll_scrim()
        self.assertIn("中止しました", self.app.scrim_result.get())
        self.assertIsNone(self.app.scrim_job)

    def test_failed_team_selection_save_preserves_previous_selection(self):
        self.app.name_input.insert("1.0", ",".join(PLAYERS))
        self.app.register_players()
        self.register_team("チームA", PLAYERS[:5])
        self.app.show_preparation()
        self.app.prep_team_choice.set("チームA")
        with patch.object(self.store, "save", side_effect=OSError("disk unavailable")), patch("run_realtime_season.messagebox.showerror"):
            self.app.confirm_preparation()
        self.assertIsNone(self.app.state.selected_team)
        self.assertIsNone(self.store.load_or_create().selected_team)

    def test_name_input_lineup_replacement_and_screen_restoration(self):
        self.assertEqual(str(self.app.confirm_button["state"]), "disabled")
        self.app.name_input.insert("1.0", "Leo,Boaster\nDerke、Chronicle，Alfajer\nMeiy")
        self.app.register_players()
        self.assertEqual(len(self.store.load_or_create().owned_players), 6)
        self.assertEqual(self.app.name_input.get("1.0", "end").strip(), "")
        for name in PLAYERS[:5]:
            self.select_inventory(name)
            self.app.add_player()
        self.assertEqual(str(self.app.confirm_button["state"]), "normal")
        self.select_inventory("Meiy")
        self.app.add_player()
        self.assertEqual(self.app.state.roster, PLAYERS[:5])
        self.app.roster.selection_set("4")
        self.app.remove_player()
        self.assertIsNotNone(self.app.state.player("Alfajer"))
        self.select_inventory("Meiy")
        self.app.add_player()
        self.app.preset_name.set("日本語チーム")
        self.app.confirm()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.preset_name, "日本語チーム")
        self.assertEqual(loaded.team_name, "マイチーム")
        self.assertEqual(loaded.roster, (*PLAYERS[:4], "Meiy"))
        self.root.destroy()
        self.root = tk.Tk()
        self.root.withdraw()
        restored = RealtimeSeasonApp(self.root, self.store, loaded)
        self.assertEqual(restored.preset_name.get(), "日本語チーム")
        self.assertEqual(restored.roster.item("4", "values")[1], "Meiy")
        self.assertEqual(len(restored.players.get_children()), 6)

    def test_failed_save_does_not_change_screen_state(self):
        original = self.app.state
        self.app.name_input.insert("1.0", "Leo")
        with patch.object(self.store, "save", side_effect=OSError("disk unavailable")), patch("run_realtime_season.messagebox.showerror"):
            self.app.register_players()
        self.assertEqual(self.app.state, original)
        self.assertEqual(len(self.app.players.get_children()), 0)
        self.assertEqual(self.app.name_input.get("1.0", "end").strip(), "Leo")
        self.assertEqual(self.store.load_or_create(), original)

    def test_search_role_filter_and_owned_removal(self):
        self.app.name_input.insert("1.0", ",".join(PLAYERS))
        self.app.register_players()
        self.app.search.set("LEO")
        self.assertEqual(len(self.app.players.get_children()), 1)
        self.assertIn("シーカー", self.app.role_filter["values"])
        self.app.role.set("スモーカー")
        self.assertEqual(len(self.app.players.get_children()), 0)
        self.app.search.set("")
        self.assertEqual(len(self.app.players.get_children()), 1)
        self.select_inventory("Boaster")
        self.app.delete_owned_player()
        self.assertIsNone(self.store.load_or_create().player("Boaster"))
        self.assertEqual(self.app.role.get(), "すべて")

    def test_unknown_name_preserves_input_and_inventory(self):
        self.app.name_input.insert("1.0", "Leo, NoSuchPlayer")
        self.app.register_players()
        self.assertEqual(self.app.state.owned_players, ())
        self.assertIn("見つかりません", self.app.status.get())
        self.assertIn("NoSuchPlayer", self.app.name_input.get("1.0", "end"))


if __name__ == "__main__":
    unittest.main()
