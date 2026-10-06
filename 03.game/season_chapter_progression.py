"""One persistent career, with separate rival and tournament contexts per chapter."""

from dataclasses import dataclass, replace

from season_competitions import (
    CompetitionTeam, SeriesScore, TournamentProgress, configured_calendar,
    definition_from_dict, parse_date,
)
from season_leagues import configured_leagues, validate_chapter
from season_loyalty import remember_loyalties
from season_transfers import TransferOffer, resolve_club_memberships, with_randomized_clubs
from season_world_levels import world_level_from_save


@dataclass(frozen=True)
class ChapterEnvironment:
    chapter: int
    opponent_teams: tuple = ()
    tournament_definitions: tuple = ()
    tournaments: tuple = ()
    rated_results: tuple = ()
    transfer_offers: tuple = ()
    chapter_started_on: str | None = None

    @classmethod
    def capture(cls, state):
        return cls(state.chapter, state.opponent_teams, state.tournament_definitions,
                   state.tournaments, state.rated_results, state.transfer_offers,
                   state.chapter_started_on)


def environment_from_save(row):
    from realtime_season import PlayerContract, SeasonClub, player_from_save

    values = dict(row)
    clubs = []
    for row in values["opponent_teams"]:
        club = dict(row)
        club["players"] = tuple(player_from_save(p) for p in club["players"])
        club["contracts"] = tuple(PlayerContract(**c) for c in club["contracts"])
        for key in ("preferred_roles", "regular_members", "acquired_members", "initial_shared_members"):
            club[key] = tuple(club[key])
        if club.get("world_level_lock") is not None:
            club["world_level_lock"] = world_level_from_save(club["world_level_lock"])
        clubs.append(SeasonClub(**club))
    values["opponent_teams"] = tuple(clubs)
    values["tournament_definitions"] = tuple(definition_from_dict(d) for d in values["tournament_definitions"])
    runs = []
    for row in values["tournaments"]:
        run = dict(row)
        run["entrants"] = tuple(CompetitionTeam(**{
            **team, "players": tuple(player_from_save(p) for p in team["players"])
        }) for team in run["entrants"])
        run["results"] = tuple(SeriesScore(**s) for s in run["results"])
        run["ranking"] = tuple(run["ranking"])
        runs.append(TournamentProgress(**run))
    values["tournaments"] = tuple(runs)
    values["rated_results"] = tuple(values["rated_results"])
    values["transfer_offers"] = tuple(TransferOffer(**o) for o in values["transfer_offers"])
    return ChapterEnvironment(**values)


def is_furina_cup(event):
    # Annual IDs stay stable across years; old fixed-date saves used the name.
    if event.annual_id is not None:
        return event.annual_id == "furina"
    return event.id == "furina" or event.name == "フリーナ杯"


def unlock_after_championship(state):
    won = any(run.completed and not run.declined and run.own_team_id == state.club_id
              and run.ranking and run.ranking[0] == state.club_id
              and is_furina_cup(state.tournament_definition(run.tournament_id))
              for run in state.tournaments)
    following = next((c for c in configured_leagues() if c > state.chapter), None)
    if not won or following is None or following in state.unlocked_chapters:
        return state
    return replace(state, unlocked_chapters=tuple(sorted((*state.unlocked_chapters, following))))


def retained_offers(state, environment):
    owned = {p.name for p in state.owned_players}
    club_ids = {c.id for c in environment.opponent_teams}
    return tuple(replace(o, status="cancelled")
                 if o.status in ("pending", "rejected") and
                 (o.player_name not in owned or o.team_id not in club_ids) else o
                 for o in environment.transfer_offers)


def saved_chapter_start(state, chapter):
    if chapter == 1:
        return None
    marker = f"→第{chapter}章（"
    for entry in state.history:
        if entry.get("種別", "").startswith("章移動: ") and marker in entry["種別"]:
            return entry["ゲーム内日付"]
    # Without an arrival record, do not retroactively start old events.
    return state.date.isoformat()


def without_prearrival_tournaments(state):
    """Past events in a newly entered chapter were never held."""
    start = parse_date(state.chapter_started_on or state.start_date)
    own_ids = {r.tournament_id for r in state.tournaments if r.own_team_id is not None}
    definitions = tuple(e for e in state.tournament_definitions
                        if parse_date(e.start_date) >= start or e.id in own_ids)
    if definitions == state.tournament_definitions:
        return state
    ids = {e.id for e in definitions}
    runs = tuple(r for r in state.tournaments if r.tournament_id in ids)
    return replace(state, tournament_definitions=definitions,
                   tournaments=runs,
                   world_level_lock=state.world_level_lock if any(not r.completed for r in runs) else None)


def enter_chapter(state, chapter):
    from realtime_season import SeasonSaveError, configured_season_teams
    from season_contract_endings import settle_contract_endings
    from season_pair_familiarity import initial_pair_days

    validate_chapter(chapter)
    if chapter not in state.unlocked_chapters:
        raise SeasonSaveError("前の章のフリーナ杯で優勝すると、この章が解禁されます。")
    if chapter == state.chapter:
        return state
    if state.starter_selection_pending:
        raise SeasonSaveError("第1章の初期選手を選んでから章を変更してください。")
    if state.day_advance_pending:
        raise SeasonSaveError("完了したアクションの日付進行を済ませてから章を変更してください。")
    if any(not run.completed for run in state.tournaments):
        raise SeasonSaveError("大会終了後に章を変更してください。")

    state = remember_loyalties(state)
    archives = {a.chapter: a for a in state.chapter_environments}
    archives[state.chapter] = ChapterEnvironment.capture(state)
    destination = archives.pop(chapter, None)
    snapshots = {p.name: p for p in state.developed_players}
    snapshots.update((p.name, p) for c in state.opponent_teams for p in c.players)
    snapshots.update((p.name, p) for p in state.owned_players)
    candidate = replace(state, chapter=chapter, opponent_teams=(), tournaments=(),
                        tournament_definitions=(), transfer_offers=(), rated_results=(),
                        world_level_lock=None, developed_players=tuple(snapshots.values()),
                        chapter_environments=tuple(archives[k] for k in sorted(archives)),
                        chapter_started_on=(state.date.isoformat() if destination is None
                                            else destination.chapter_started_on))
    if destination is None:
        clubs = configured_season_teams(
            transferred_players=state.transferred_players, game_month=state.game_month,
            start_date=state.start_date, game_date=state.date.isoformat(),
            priority_names=tuple(p.name for p in state.owned_players),
            loyalty_memory=state.team_loyalties, contract_memory=state.contract_signings,
            refusal_memory=state.contract_bans, chapter=chapter)
        candidate = replace(candidate, opponent_teams=clubs)
        # Use saved growth when players first appear in another chapter.
        candidate = replace(candidate, opponent_teams=tuple(replace(c, players=tuple(
            snapshots.get(p.name, p) for p in c.players)) for c in clubs))
        candidate = with_randomized_clubs(candidate)
        candidate = initial_pair_days(candidate)
        protected = ()
    else:
        clubs = tuple(replace(c, players=tuple(snapshots.get(p.name, p) for p in c.players),
                              world_level_lock=None) for c in destination.opponent_teams)
        clubs = resolve_club_memberships(clubs, tuple(p.name for p in state.owned_players), existing_teams=clubs)
        candidate = replace(candidate, opponent_teams=clubs, tournaments=destination.tournaments,
                            rated_results=destination.rated_results,
                            transfer_offers=retained_offers(state, destination))
        protected_ids = {r.tournament_id for r in destination.tournaments}
        protected = tuple(e for e in destination.tournament_definitions if e.id in protected_ids)
    _, periods, definitions = configured_calendar(on_date=state.date, existing_definitions=protected)
    protected_ids = {e.id for e in protected}
    # A new league does not retroactively hold events before arrival.
    definitions = tuple(e for e in definitions if e.id not in protected_ids and parse_date(e.start_date) >= state.date)
    candidate = replace(candidate, in_season_periods=periods,
                        tournament_definitions=(*protected, *definitions)).with_registered_ratings()
    candidate = without_prearrival_tournaments(candidate)
    candidate = settle_contract_endings(candidate)
    candidate = remember_loyalties(candidate)
    candidate.validate()
    return candidate._record_history(f"章移動: 第{state.chapter}章→第{chapter}章（{candidate.league_name}）")


def validate_progression(state):
    if (not isinstance(state.unlocked_chapters, tuple) or 1 not in state.unlocked_chapters
            or any(type(c) is not int or c not in configured_leagues() for c in state.unlocked_chapters)
            or state.unlocked_chapters != tuple(sorted(set(state.unlocked_chapters)))):
        raise ValueError("章の解禁データが不正です。")
    if not isinstance(state.chapter_environments, tuple):
        raise ValueError("章別の進行データが不正です。")
    seen = {state.chapter}
    for environment in state.chapter_environments:
        if (not isinstance(environment, ChapterEnvironment) or environment.chapter in seen
                or environment.chapter not in state.unlocked_chapters):
            raise ValueError("章別の進行データが重複または不正です。")
        seen.add(environment.chapter)
        if any(not r.completed for r in environment.tournaments):
            raise ValueError("未完了の大会がある章は保存先を切り替えられません。")
        # Validate archived rosters, finances, contracts and completed brackets
        # without recursively validating the same archives.
        replace(state, chapter=environment.chapter, chapter_environments=(),
                chapter_started_on=environment.chapter_started_on,
                opponent_teams=resolve_club_memberships(environment.opponent_teams,
                    tuple(p.name for p in state.owned_players), existing_teams=environment.opponent_teams),
                tournament_definitions=environment.tournament_definitions,
                tournaments=environment.tournaments, rated_results=environment.rated_results,
                transfer_offers=retained_offers(state, environment),
                world_level_lock=None).validate()
