"""Persistent player inventory and lineup for the real-time season mode."""

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import json
import math
import os
from pathlib import Path
from random import Random
import tempfile
from uuid import NAMESPACE_URL, uuid4, uuid5

from character_stats import CharacterStats, all_characters, get_by_name
from season_ratings import DEFAULT_TEAM_RATING, SeasonRating, expected_score, series_ratings
from season_monthly_events import MonthlyEvent, process_monthly_events
from season_salary import (SalaryMode, SalaryRecord, SalarySettings, SalaryDataError,
                           configured_salary_settings, salary_records, log_salary_change, clamp_change)
from season_competitions import (
    CompetitionError, CompetitionTeam, SeriesScore, TournamentProgress, add_months,
    configured_calendar, definition_from_dict, month_index, next_match, parse_date,
    phase_for, player_eliminated, validate_definitions, validate_periods,
)


SAVE_VERSION = 14
PLAYER_CLUB_ID = "player_club"
ROSTER_SIZE = 5
INITIAL_MONEY = 10_000_000
MIN_MONTHLY_SPONSOR_INCOME = 5_000_000
CONTRACT_OPTIONS = {"短期契約": "short", "1年契約": "year1", "2年契約": "year2", "3年契約": "year3"}
DEFAULT_SAVE_PATH = Path(__file__).resolve().parent / "data" / "realtime_season" / "save.json"


class SeasonSaveError(ValueError):
    """Invalid or unsupported season data; never reset it implicitly."""


@dataclass(frozen=True)
class ContractTerms:
    kind: str
    months: int
    monthly_salary: int

    @property
    def required_funds(self):
        return self.monthly_salary * self.months

    @property
    def signing_bonus(self):
        return self.monthly_salary * 3

    @property
    def total_required_funds(self):
        return self.signing_bonus + self.required_funds


def contract_terms(player, kind, short_months=6):
    validate_player(player)
    if kind == "short":
        if type(short_months) is not int or not 1 <= short_months <= 6:
            raise SeasonSaveError("短期契約の期間は1～6か月から選択してください。")
        return ContractTerms(kind, short_months, player.monthly_salary)
    terms = {"year1": (12, 10), "year2": (24, 9), "year3": (36, 8)}
    if not isinstance(kind, str) or kind not in terms:
        raise SeasonSaveError("契約の種類を選択してください。")
    if player.loyalty == 0:
        raise SeasonSaveError("忠誠心が0の選手とは短期契約のみ結べます。")
    months, tenths = terms[kind]
    # 円単位で保存し、割引後の端数は切り上げる。
    return ContractTerms(kind, months, (player.monthly_salary * tenths + 9) // 10)


@dataclass(frozen=True)
class PlayerContract:
    player_name: str
    kind: str
    monthly_salary: int
    start_month: int
    duration_months: int
    team_loyalty: float = 50.0
    end_reason: str | None = None

    @property
    def end_month(self):
        return self.start_month + self.duration_months

    def active(self, month):
        return self.end_reason is None and self.start_month <= month < self.end_month

    def elapsed(self, month):
        return max(0, min(month - self.start_month, self.duration_months))

    def remaining(self, month):
        return max(0, self.end_month - month) if self.end_reason is None else 0


def initial_contract(player, month=0):
    if player.loyalty == 0:
        return PlayerContract(player.name, "short", player.monthly_salary, month, 6)
    return PlayerContract(player.name, "year1", player.monthly_salary, month, 12)


def player_from_save(row, *, legacy_loyalty=False):
    """Add catalog economy settings to old snapshots without resetting abilities."""
    if not isinstance(row, dict):
        raise SeasonSaveError("選手のデータ形式が不正です。")
    row = dict(row)
    # Only saved values use the old scale; missing values come from today's catalog.
    if legacy_loyalty and "loyalty" in row:
        value = row["loyalty"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise SeasonSaveError("保存された忠誠心が不正です。")
        row["loyalty"] = value / 10
    catalog = get_by_name(row.get("name")) if isinstance(row.get("name"), str) else None
    if catalog:
        row.setdefault("monthly_salary", catalog.monthly_salary)
        row.setdefault("loyalty", catalog.loyalty)
    return CharacterStats(**row)


@dataclass(frozen=True)
class SeasonTeam:
    """A five-player lineup preset within the player's single club."""
    id: str
    name: str
    roster: tuple[str, ...]
    igl: str | None = None
    carrier: str | None = None
    ai: str = "default"


@dataclass(frozen=True)
class SeasonClub:
    id: str
    name: str
    players: tuple[CharacterStats, ...]
    igl: str | None = None
    carrier: str | None = None
    ai: str = "default"
    transfer_multiplier: float = 12.0
    preferred_roles: tuple[str, ...] = ()
    contracts: tuple[PlayerContract, ...] = ()

    @property
    def members(self):
        return tuple(player.name for player in self.players)

    @property
    def roster(self):
        return self.members[:ROSTER_SIZE]

    @property
    def effective_igl(self):
        return self.igl if self.igl is not None else (max(self.players[:ROSTER_SIZE], key=lambda p: p.iq).name if self.players else None)

    @property
    def effective_carrier(self):
        return self.carrier if self.carrier is not None else (self.players[0].name if self.players else None)


def validate_preset_settings(roster, igl, carrier, ai):
    from roster_select import TEAM_AI_OPTIONS
    for label, name in (("IGL", igl), ("キャリアー", carrier)):
        if name is not None and (not isinstance(name, str) or name not in roster):
            raise SeasonSaveError(f"{label}はこのプリセットのロスターから選択してください。")
    if not isinstance(ai, str) or ai not in TEAM_AI_OPTIONS.values():
        raise SeasonSaveError("プリセットのAIが不正です。")


def validate_player(player):
    if not isinstance(player, CharacterStats):
        raise SeasonSaveError("選手のデータ形式が不正です。")
    if not isinstance(player.name, str) or not player.name.strip():
        raise SeasonSaveError("選手名が不正です。")
    if not isinstance(player.role, str) or not player.role.strip():
        raise SeasonSaveError(f"{player.name}のロールが不正です。")
    for field in fields(CharacterStats):
        key = field.name
        if key in {"name", "role"}:
            continue
        value = getattr(player, key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise SeasonSaveError(f"{player.name}の能力値 {key} が不正です。")
        if key == "monthly_salary" and type(value) is not int:
            raise SeasonSaveError(f"{player.name}の月給は円単位の整数で設定してください。")
        maximum = 1 if key in {"hs_pct", "dodge_pct", "hit_pct"} else (
            10 if key in {"form_variance", "mental", "loyalty"} else None
        )
        if maximum is not None and value > maximum:
            raise SeasonSaveError(f"{player.name}の能力値 {key} が範囲外です。")


@dataclass(frozen=True)
class SeasonState:
    team_name: str
    owned_players: tuple[CharacterStats, ...]
    roster: tuple[str, ...] = ()
    teams: tuple[SeasonTeam, ...] = ()
    editing_team_id: str | None = None
    selected_team_id: str | None = None
    opponent_teams: tuple[SeasonClub, ...] = ()
    money: int = INITIAL_MONEY
    game_month: int = 0
    contracts: tuple[PlayerContract, ...] = ()
    start_date: str = "2026-01-01"
    game_date: str | None = None
    in_season_periods: tuple[tuple[str, str], ...] = (("03-01", "11-30"),)
    tournament_definitions: tuple = ()
    tournaments: tuple[TournamentProgress, ...] = ()
    starter_candidates: tuple[CharacterStats, ...] = ()
    starter_selection: tuple[str, ...] = ()
    starter_selection_pending: bool = False
    ratings: tuple[SeasonRating, ...] = ()
    rated_results: tuple[str, ...] = ()
    sponsor_active: bool = True
    transferred_players: tuple[str, ...] = ()
    monthly_events: tuple[MonthlyEvent, ...] = ()
    monthly_events_through: int = 0
    preset_name: str = "編成1"
    club_id: str = PLAYER_CLUB_ID
    preset_igl: str | None = None
    preset_carrier: str | None = None
    preset_ai: str = "default"
    salary_mode: SalaryMode = SalaryMode.STATIC
    salary_settings: SalarySettings | None = None
    salary_records: tuple[SalaryRecord, ...] = ()
    salary_updated_month: int | None = None

    def salary_player(self, player):
        if self.salary_mode == SalaryMode.STATIC:
            return player
        record = next((r for r in self.salary_records if r.name == player.name), None)
        return replace(player, monthly_salary=record.monthly_salary) if record else player

    def contract_terms(self, player, kind, short_months=6):
        terms = contract_terms(player, kind, short_months)
        record = next((r for r in self.salary_records if r.name == player.name), None)
        if self.salary_mode == SalaryMode.KD_DYNAMIC and record is not None and record.fixed:
            return replace(terms, monthly_salary=record.monthly_salary)
        return terms

    def with_salary_mode(self, mode):
        if not self.starter_selection_pending:
            raise SeasonSaveError("給与モードは新規シーズン開始前だけ選択できます。")
        try:
            mode = SalaryMode(mode)
            if mode == self.salary_mode:
                return self
            if mode == SalaryMode.KD_DYNAMIC:
                return replace(self, salary_mode=mode, salary_settings=configured_salary_settings()).with_updated_salaries(initial=True)
            # Switching before start restores the original static snapshots.
            static = {r.name: r.static_salary for r in self.salary_records}
            def restore(player):
                return replace(player, monthly_salary=static.get(player.name, player.monthly_salary))
            clubs = []
            for club in self.opponent_teams:
                players = tuple(restore(p) for p in club.players)
                by_name = {p.name: p for p in players}
                contracts = tuple(replace(c, monthly_salary=contract_terms(by_name[c.player_name], c.kind, c.duration_months).monthly_salary)
                                  if c.active(self.game_month) else c for c in club.contracts)
                clubs.append(replace(club, players=players, contracts=contracts))
            candidate = replace(self, salary_mode=mode, salary_settings=None, salary_records=(), salary_updated_month=None,
                                starter_candidates=tuple(restore(p) for p in self.starter_candidates), opponent_teams=tuple(clubs))
            candidate.validate()
            return candidate
        except SalaryDataError as exc:
            raise SeasonSaveError(str(exc)) from exc

    def with_updated_salaries(self, *, initial=False):
        if self.salary_mode == SalaryMode.STATIC or not initial and self.salary_updated_month == self.game_month:
            return self
        try:
            pool = {p.name: p for p in all_characters()}
            pool.update((p.name, p) for club in self.opponent_teams for p in club.players)
            pool.update((p.name, p) for p in (*self.starter_candidates, *self.owned_players))
            old = {r.name: r for r in self.salary_records}
            static = {name: p.monthly_salary for name, p in pool.items()}
            static.update((name, r.static_salary) for name, r in old.items())
            previous = None if initial else {name: old[name].monthly_salary if name in old else p.monthly_salary
                                            for name, p in pool.items()}
            if previous is not None:
                previous.update((name, r.monthly_salary) for name, r in old.items())
            records = salary_records(static, self.salary_settings, previous)
        except (SalaryDataError, OSError) as exc:
            raise SeasonSaveError(str(exc)) from exc
        by_name = {r.name: r for r in records}
        def update_player(p):
            return replace(p, monthly_salary=by_name[p.name].monthly_salary)
        changes = []
        def update_contract(c):
            record = by_name.get(c.player_name)
            if record is None or not c.active(self.game_month):
                return c
            tenths = {"short": 10, "year1": 10, "year2": 9, "year3": 8}[c.kind]
            salary = record.monthly_salary if record.fixed else (record.monthly_salary * tenths + 9) // 10
            if not initial and not record.fixed:
                salary = clamp_change(salary, c.monthly_salary, self.salary_settings.max_change)
            changes.append((c, salary, record))
            return replace(c, monthly_salary=salary)
        clubs = tuple(replace(club, players=tuple(update_player(p) for p in club.players),
                              contracts=tuple(update_contract(c) for c in club.contracts)) for club in self.opponent_teams)
        candidate = replace(self, salary_records=records, salary_updated_month=self.game_month,
                            owned_players=tuple(update_player(p) for p in self.owned_players),
                            starter_candidates=tuple(update_player(p) for p in self.starter_candidates),
                            contracts=tuple(update_contract(c) for c in self.contracts), opponent_teams=clubs)
        candidate.validate()
        for r in records:
            before = old[r.name].monthly_salary if r.name in old else r.static_salary
            log_salary_change(r.name, before, r.monthly_salary, r)
        for c, salary, r in changes:
            log_salary_change(c.player_name, c.monthly_salary, salary, r, contract=True)
        return candidate

    def with_registered_ratings(self):
        records = {r.team_id: r for r in self.ratings}
        for run in self.tournaments:
            for team in run.entrants:
                records.setdefault(team.id, SeasonRating(team.id, team.name))
        for team in (SeasonTeam(self.club_id, self.team_name, ()), *self.opponent_teams):
            old = records.get(team.id)
            records[team.id] = SeasonRating(team.id, team.name, old.value if old else DEFAULT_TEAM_RATING)
        return replace(self, ratings=tuple(records[key] for key in sorted(records)))

    def rating(self, team_id):
        if self.team(team_id) is not None:
            team_id = self.club_id
        return next((r.value for r in self.ratings if r.team_id == team_id), DEFAULT_TEAM_RATING)

    @property
    def rating_ranking(self):
        active_ids = {self.club_id, *(t.id for t in self.opponent_teams)}
        return tuple(sorted((r for r in self.with_registered_ratings().ratings if r.team_id in active_ids),
                            key=lambda r: (-r.value, r.team_name.casefold())))

    def with_rated_result(self, result_id, left_id, right_id, left_wins, right_wins):
        if not isinstance(result_id, str) or not result_id:
            raise SeasonSaveError("レート更新の試合IDが不正です。")
        if result_id in self.rated_results:
            return self
        state = self.with_registered_ratings()
        left_id = state.club_id if state.team(left_id) is not None else left_id
        right_id = state.club_id if state.team(right_id) is not None else right_id
        ids = {r.team_id for r in state.ratings}
        if left_id == right_id or left_id not in ids or right_id not in ids:
            raise SeasonSaveError("レート更新の対戦チームが不正です。")
        if (type(left_wins) is not int or type(right_wins) is not int or min(left_wins, right_wins) < 0
                or left_wins == right_wins):
            raise SeasonSaveError("レート更新には勝敗が確定したマップ数が必要です。")
        left, right = series_ratings(state.rating(left_id), state.rating(right_id), left_wins, right_wins)
        state = replace(state, ratings=tuple(replace(r, value=left if r.team_id == left_id else right)
                                              if r.team_id in (left_id, right_id) else r for r in state.ratings),
                        rated_results=(*state.rated_results, result_id))
        state = state._with_match_loyalty(left_id, right_id, left_wins > right_wins)
        state.validate()
        return state

    def _with_match_loyalty(self, left_id, right_id, left_won):
        outcomes = {left_id: left_won, right_id: not left_won}

        def updated_contracts(team_id, players, contracts):
            if team_id not in outcomes:
                return contracts
            members = {p.name: p for p in players}
            updated = []
            for contract in contracts:
                player = members.get(contract.player_name)
                if player is None or not contract.active(self.game_month):
                    updated.append(contract)
                    continue
                delta = 1.0 if outcomes[team_id] else -(10 - player.loyalty) / 10
                updated.append(replace(contract, team_loyalty=round(contract.team_loyalty + delta, 10))
                               if delta else contract)
            return tuple(updated)

        return replace(self,
            contracts=updated_contracts(self.club_id, self.owned_players, self.contracts),
            opponent_teams=tuple(replace(club, contracts=updated_contracts(club.id, club.players, club.contracts))
                                 if club.id in outcomes else club for club in self.opponent_teams))

    @property
    def sponsor_team(self):
        if self.starter_selection_pending:
            return None
        preset = self.selected_team or (self.teams[0] if self.teams else None)
        return SeasonTeam(self.club_id, self.team_name, preset.roster if preset else ())

    @property
    def monthly_sponsor_income(self):
        team = self.sponsor_team
        if not self.sponsor_active or team is None:
            return 0
        return max(MIN_MONTHLY_SPONSOR_INCOME, int(Decimal(str(self.rating(team.id))) * 5000))

    @property
    def monthly_payroll(self):
        return sum(c.monthly_salary for c in self.contracts if c.active(self.game_month))

    def with_sponsor_contract(self, active):
        if type(active) is not bool:
            raise SeasonSaveError("スポンサー契約の状態が不正です。")
        candidate = replace(self, sponsor_active=active)
        candidate.validate()
        return candidate

    @property
    def scout_players(self):
        # Saved ability and salary snapshots take priority over the live catalog.
        players = {p.name: p for p in all_characters()}
        players.update((p.name, p) for club in self.opponent_teams for p in club.players)
        players.update((p.name, p) for p in self.owned_players)
        return tuple(self.salary_player(p) for p in players.values())

    def player_affiliation(self, name):
        own = self.player(name)
        if own is not None:
            return self.team_name
        owner = self.opponent_owner(name)
        return owner.name if owner else "LFT"

    def transfer_fee(self, name):
        owner = self.opponent_owner(name)
        if owner is None:
            return 0
        player = next(p for p in owner.players if p.name == name)
        return int((Decimal(player.monthly_salary) * Decimal(str(owner.transfer_multiplier))).to_integral_value(rounding=ROUND_CEILING))

    def recruitment_blocked(self, name):
        if self.player(name) is not None:
            return "すでに所持している選手です。再契約は契約状況から行ってください。"
        if any(not run.completed and any(p.name == name for t in run.entrants for p in t.players) for run in self.tournaments):
            return "大会参加中の選手は引き抜けません。大会終了後に契約してください。"
        return ""

    def with_starter_selection(self, names):
        if not self.starter_selection_pending:
            raise SeasonSaveError("初期キャラの選択はすでに終了しています。")
        candidate = replace(self, starter_selection=tuple(names))
        candidate.validate()
        return candidate

    def with_initial_selection(self, names=None):
        if not self.starter_selection_pending:
            raise SeasonSaveError("初期キャラはすでに入手済みです。")
        names = self.starter_selection if names is None else tuple(names)
        if len(names) != ROSTER_SIZE or any(not isinstance(name, str) for name in names) or len(set(names)) != ROSTER_SIZE:
            raise SeasonSaveError("初期キャラは異なる5人を選んでください。")
        pool = {player.name: player for player in self.starter_candidates}
        if not set(names).issubset(pool):
            raise SeasonSaveError("初期キャラの候補から選んでください。")
        players = tuple(pool[name] for name in names)
        candidate = replace(self, starter_selection_pending=False, starter_selection=tuple(names),
                            owned_players=players, contracts=tuple(
                                PlayerContract(p.name, "short", p.monthly_salary, self.game_month, 6) for p in players))
        candidate.validate()
        return candidate.with_updated_salaries(initial=True)

    @property
    def date(self):
        return parse_date(self.game_date) if self.game_date is not None else add_months(parse_date(self.start_date), self.game_month)

    @property
    def phase(self):
        return phase_for(self.date, self.in_season_periods)

    def tournament(self, event_id):
        return next((run for run in self.tournaments if run.tournament_id == event_id), None)

    def tournament_definition(self, event_id):
        return next((event for event in self.tournament_definitions if event.id == event_id), None)

    @property
    def visible_tournaments(self):
        return tuple(event for event in self.tournament_definitions if self.tournament(event.id) is not None or event.appears(self))

    @property
    def pending_tournaments(self):
        return tuple(event for event in self.visible_tournaments
                     if parse_date(event.start_date) <= self.date
                     and (self.tournament(event.id) is None and event.allow_player_entry and self.date <= parse_date(event.end_date)
                          or self.tournament(event.id) is not None and not self.tournament(event.id).completed))

    def check_tournament_match_day(self, event_id):
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        if event is None or run is None or run.completed:
            raise SeasonSaveError("進行中の大会を選択してください。")
        if self.date < parse_date(event.start_date):
            raise SeasonSaveError("大会試合は開始日以降に行ってください。")
        if run.last_match_date is not None and self.date <= parse_date(run.last_match_date):
            raise SeasonSaveError("大会は1日1試合です。1日進めてから次のシリーズを開始してください。")

    def with_tournament_entry(self, event_id, team_id=None, *, own_ai=None, igl=None, carrier=None):
        event = self.tournament_definition(event_id)
        if event is None or not event.appears(self) or self.date > parse_date(event.end_date):
            raise SeasonSaveError("大会は未出現または参加登録の締切を過ぎています。")
        if self.tournament(event_id) is not None:
            raise SeasonSaveError("この大会は参加判断済みです。")
        own = self.team(team_id) if event.allow_player_entry else None
        if event.allow_player_entry and (own is None or any(not self.can_play(name) for name in own.roster)):
            raise SeasonSaveError("契約中の5人を編成した参加チームを選択してください。")
        if own and any(not r.completed and r.own_team_id is not None for r in self.tournaments):
            raise SeasonSaveError("参加中の大会を完了してから次の大会へ参加してください。")
        entrants = []
        if own:
            players = tuple(self.player(name) for name in own.roster)
            entrants.append(CompetitionTeam(self.club_id, self.team_name, players, own.ai if own_ai is None else own_ai,
                igl or own.igl or max(players, key=lambda p: p.iq).name, carrier or own.carrier or players[0].name))
        rivals = self.opponent_teams
        if event.opponent_teams:
            by_name = {club.name: club for club in rivals}
            missing = set(event.opponent_teams) - set(by_name)
            if missing:
                raise SeasonSaveError(f"大会の相手チームが見つかりません: {', '.join(sorted(missing))}")
            rivals = tuple(by_name[name] for name in event.opponent_teams)
        rivals = tuple(club for club in rivals if len(club.players) >= ROSTER_SIZE)
        needed = event.team_count - len(entrants)
        if len(rivals) < needed:
            raise SeasonSaveError(f"大会には他チームが{needed}チーム必要です（現在{len(rivals)}）。所属設定を追加してください。")
        entrants.extend(CompetitionTeam(club.id, club.name, club.players[:ROSTER_SIZE], club.ai,
                                       club.effective_igl, club.effective_carrier) for club in rivals[:needed])
        run = TournamentProgress(event.id, self.club_id if own else None, tuple(entrants),
            seed=int(uuid4().hex[:8], 16) % (2**31), preset_id=own.id if own else None)
        candidate = replace(self, tournaments=(*self.tournaments, run))
        candidate.validate()
        return candidate

    def with_declined_tournament(self, event_id):
        event = self.tournament_definition(event_id)
        if event is None or not event.appears(self) or self.date > parse_date(event.end_date):
            raise SeasonSaveError("大会は未出現または参加登録の締切を過ぎています。")
        if not event.participation_optional or not event.allow_player_entry:
            raise SeasonSaveError("この大会では不参加を選択できません。")
        if self.tournament(event_id) is not None:
            raise SeasonSaveError("この大会は参加判断済みです。")
        candidate = replace(self, tournaments=(*self.tournaments, TournamentProgress(event_id, None, completed=True, declined=True)))
        candidate.validate()
        return candidate

    def with_tournament_result(self, event_id, score):
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        if event is None or run is None or run.completed:
            raise SeasonSaveError("結果を登録する進行中の大会が見つかりません。")
        self.check_tournament_match_day(event_id)
        candidate = self._with_tournament_score(event, run, score)
        updated = candidate.tournament(event_id)
        if not updated.completed and player_eliminated(event, updated):
            return candidate.with_tournament_rating_finish(event_id)
        return candidate

    def _with_tournament_score(self, event, run, score):
        """Record one series, including its rating update and completion prize."""
        event_id = event.id
        updated = replace(run, results=(*run.results, score), last_match_date=self.date.isoformat())
        pending, ranking = next_match(event, updated)
        prize = 0
        if pending is None:
            if run.own_team_id is not None:
                rank = ranking.index(run.own_team_id) + 1
                prize = event.prizes.get(rank, 0)
            updated = replace(updated, completed=True, ranking=ranking, prize_paid=prize, completed_date=self.date.isoformat())
        candidate = replace(self, money=self.money + prize,
                            tournaments=tuple(updated if r.tournament_id == event_id else r for r in self.tournaments))
        candidate.validate()
        candidate = candidate.with_rated_result(f"tournament:{event_id}:{score.match_id}", score.left_id, score.right_id, score.left_wins, score.right_wins)
        return candidate

    def _tournament_rating_score(self, run, match):
        if run.own_team_id in (match.left, match.right):
            raise SeasonSaveError("自チームの試合をレート判定することはできません。")
        probability = expected_score(self.rating(match.left), self.rating(match.right))
        # Stable across save/reload and failed save retries; one draw per series.
        draw = Random((run.seed + len(run.results) * 1000) % (2**31)).random()
        left_wins = draw < probability
        return SeriesScore(match.id, match.left, match.right,
                           match.maps_to_win if left_wins else 0,
                           0 if left_wins else match.maps_to_win, decided_by_rating=True)

    def with_tournament_rating_result(self, event_id):
        """Resolve today's NPC series using the current Elo win probability."""
        self.check_tournament_match_day(event_id)
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        match, _ = next_match(event, run)
        return self.with_tournament_result(event_id, self._tournament_rating_score(run, match))

    def with_tournament_rating_finish(self, event_id):
        """Finish NPC cards on the current day after the player's elimination."""
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        if event is None or run is None:
            raise SeasonSaveError("進行中の大会を選択してください。")
        if run.completed:
            return self
        if not player_eliminated(event, run):
            raise SeasonSaveError("自チームの敗退が確定してからレート判定で終了できます。")
        candidate = self
        while not run.completed:
            match, _ = next_match(event, run)
            score = candidate._tournament_rating_score(run, match)
            candidate = candidate._with_tournament_score(event, run, score)
            run = candidate.tournament(event_id)
        return candidate

    def with_tournament_forfeit(self, event_id):
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        if event is None or run is None or run.completed:
            raise SeasonSaveError("進行中の大会を選択してください。")
        match, _ = next_match(event, run)
        if run.own_team_id not in (match.left, match.right):
            raise SeasonSaveError("自分のチームのシリーズだけ棄権できます。")
        score = SeriesScore(match.id, match.left, match.right,
                            0 if run.own_team_id == match.left else match.maps_to_win,
                            0 if run.own_team_id == match.right else match.maps_to_win)
        return self.with_tournament_result(event_id, score)

    def contract(self, name):
        return next((c for c in self.contracts if c.player_name == name), None)

    def can_play(self, name):
        contract = self.contract(name)
        return self.player(name) is not None and contract is not None and contract.active(self.game_month)

    @property
    def lft_players(self):
        affiliated = {p.name for p in self.owned_players}
        affiliated.update(p.name for team in self.opponent_teams for p in team.players)
        return tuple(self.salary_player(p) for p in all_characters() if p.name not in affiliated)

    def with_scouted_player(self, name, kind, short_months=6):
        blocked = self.recruitment_blocked(name)
        if blocked:
            raise SeasonSaveError(blocked)
        player = next((p for p in self.scout_players if p.name == name), None)
        if player is None:
            raise SeasonSaveError("スカウトする選手が見つかりません。")
        old = self.contract(name)
        if old is not None and old.team_loyalty <= 0:
            raise SeasonSaveError("チームへの忠誠が0以下のため、この選手とは再契約できません。")
        fee = self.transfer_fee(name)
        terms = self.contract_terms(player, kind, short_months)
        if self.money < fee + terms.total_required_funds:
            raise SeasonSaveError(f"契約には移籍金{fee:,}円・契約金{terms.signing_bonus:,}円と契約条件の資金{terms.required_funds:,}円が必要です（現在{self.money:,}円）。")
        owner = self.opponent_owner(name)
        clubs = []
        for club in self.opponent_teams:
            if club != owner:
                clubs.append(club)
                continue
            remaining = tuple(p for p in club.players if p.name != name)
            roster = {p.name for p in remaining[:ROSTER_SIZE]}
            clubs.append(replace(club, players=remaining, igl=club.igl if club.igl in roster else None,
                                 carrier=club.carrier if club.carrier in roster else None,
                                 contracts=tuple(replace(c, end_reason="released") if c.player_name == name else c for c in club.contracts)))
        candidate = replace(self, money=self.money - fee, opponent_teams=tuple(clubs),
                            transferred_players=(*self.transferred_players, name) if owner and name not in self.transferred_players else self.transferred_players)
        return candidate._with_signed_contract(player, kind, short_months, recruit=True)

    def with_renewed_contract(self, name, kind, short_months=6):
        player = self.player(name)
        old = self.contract(name)
        if player is None or old is None:
            raise SeasonSaveError("再契約する所持選手が見つかりません。")
        if old.active(self.game_month):
            raise SeasonSaveError("再契約は現在の契約が終了してから行えます。")
        if old.team_loyalty <= 0:
            raise SeasonSaveError("チームへの忠誠が0以下のため、再契約を断られました。")
        return self._with_signed_contract(player, kind, short_months, recruit=False)

    def _with_signed_contract(self, player, kind, short_months, recruit):
        terms = self.contract_terms(player, kind, short_months)
        if self.money < terms.total_required_funds:
            raise SeasonSaveError(f"契約には契約金{terms.signing_bonus:,}円と契約条件の資金{terms.required_funds:,}円が必要です（現在{self.money:,}円）。")
        previous = self.contract(player.name)
        contract = PlayerContract(player.name, kind, terms.monthly_salary, self.game_month, terms.months,
                                  previous.team_loyalty if previous else 50.0)
        candidate = replace(self,
                            money=self.money - terms.signing_bonus,
                            owned_players=(*self.owned_players, player) if recruit else self.owned_players,
                            contracts=tuple(c for c in self.contracts if c.player_name != player.name) + (contract,))
        candidate.validate()
        return candidate

    def with_team_loyalty(self, name, value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise SeasonSaveError("チームへの忠誠には有限の数値を設定してください。")
        if self.contract(name) is None:
            raise SeasonSaveError("忠誠を変更する選手の契約が見つかりません。")
        player = self.player(name)
        if player is not None and player.loyalty == 10 and value < self.contract(name).team_loyalty:
            return self
        candidate = replace(self, contracts=tuple(replace(c, team_loyalty=value) if c.player_name == name else c for c in self.contracts))
        candidate.validate()
        return candidate

    def advance_months(self, months=1, *, pay_salaries=True, stop_for_tournaments=False):
        if type(months) is not int or months < 1:
            raise SeasonSaveError("進める月数は1以上の整数で指定してください。")
        target = add_months(self.date, months)
        return self.advance_days((target - self.date).days, pay_salaries=pay_salaries, stop_for_tournaments=stop_for_tournaments)

    def advance_days(self, days=1, *, pay_salaries=True, stop_for_tournaments=True):
        if type(days) is not int or days < 1:
            raise SeasonSaveError("進める日数は1以上の整数で指定してください。")
        candidate = self
        for _ in range(days):
            pending = candidate.pending_tournaments
            # After today's series, allow exactly one day forward, then stop at
            # the next unplayed series. Month-end processing follows the same path.
            blocking = [e for e in pending if candidate.tournament(e.id) is None
                        or candidate.tournament(e.id).last_match_date != candidate.date.isoformat()]
            if blocking and (stop_for_tournaments or any(candidate.tournament(e.id) is not None or not e.participation_optional for e in blocking)):
                break
            day = candidate.date + timedelta(days=1)
            index = month_index(parse_date(candidate.start_date), day)
            new_month = index != candidate.game_month
            if index != candidate.game_month:
                income = candidate.monthly_sponsor_income
                # Decay only for the month actually spent under contract, including
                # the final month. Daily advances and bulk advances share this path.
                players = {p.name: p for p in candidate.owned_players}
                candidate = replace(candidate, contracts=tuple(
                    replace(c, team_loyalty=round(c.team_loyalty - (10 - players[c.player_name].loyalty) / 10, 10))
                    if c.active(candidate.game_month) and c.player_name in players and players[c.player_name].loyalty < 10 else c
                    for c in candidate.contracts))
                for contract in candidate.contracts:
                    if contract.kind == "short" and contract.active(candidate.game_month) and contract.team_loyalty <= 0:
                        candidate = candidate._with_departed_player(contract.player_name)
                payroll = candidate.monthly_payroll if pay_salaries else 0
                candidate = replace(candidate, money=candidate.money - payroll + income, game_month=index)
            candidate = replace(candidate, game_date=day.isoformat())
            if new_month:
                candidate = candidate.with_updated_salaries()
                candidate = process_monthly_events(candidate)
            for event in candidate.pending_tournaments:
                if not event.participation_optional and candidate.tournament(event.id) is None:
                    ready = [team for team in candidate.teams if all(candidate.can_play(name) for name in team.roster)]
                    own = candidate.selected_team if candidate.selected_team in ready else (ready[0] if ready else None)
                    if own is not None:
                        candidate = candidate.with_tournament_entry(event.id, own.id)
        candidate.validate()
        return candidate

    def _with_departed_player(self, name):
        # Registered lineups must always have five players. Keep the remaining draft
        # so the user can replace a departing player and register the team again.
        teams = tuple(t for t in self.teams if name not in t.roster)
        team_ids = {t.id for t in teams}
        return replace(self, owned_players=tuple(p for p in self.owned_players if p.name != name),
                       roster=tuple(n for n in self.roster if n != name), teams=teams,
                       preset_igl=None if self.preset_igl == name else self.preset_igl,
                       preset_carrier=None if self.preset_carrier == name else self.preset_carrier,
                       editing_team_id=self.editing_team_id if self.editing_team_id in team_ids else None,
                       selected_team_id=self.selected_team_id if self.selected_team_id in team_ids else None,
                       contracts=tuple(replace(c, end_reason="left") if c.player_name == name else c for c in self.contracts))

    @property
    def roster_ready(self):
        return len(self.roster) == ROSTER_SIZE

    def player(self, name):
        return next((p for p in self.owned_players if p.name == name), None)

    def team(self, team_id):
        return next((team for team in self.teams if team.id == team_id), None)

    @property
    def selected_team(self):
        return self.team(self.selected_team_id)

    def player_in_saved_team(self, name):
        return any(name in team.roster for team in self.teams)

    def opponent_owner(self, name):
        return next((team for team in self.opponent_teams if name in team.members), None)

    def roster_owner(self, name):
        # Presets may reuse club members; affiliation is owned_players vs rivals.
        return None

    def with_opponent_teams(self, teams):
        if any(not run.completed for run in self.tournaments):
            raise SeasonSaveError("大会参加中は他チームの所属設定を変更できません。大会終了後に取り込んでください。")
        candidate = replace(self, opponent_teams=tuple(teams)).with_registered_ratings()
        candidate.validate()
        return candidate

    def with_new_team(self):
        used_names = {team.name for team in self.teams}
        number = 1
        while f"編成{number}" in used_names:
            number += 1
        return replace(self, preset_name=f"編成{number}", roster=(), editing_team_id=None,
                       preset_igl=None, preset_carrier=None, preset_ai="default")

    def with_editing_team(self, team_id):
        team = self.team(team_id)
        if team is None:
            raise SeasonSaveError("編集するチームが見つかりません。")
        return replace(self, preset_name=team.name, roster=team.roster, editing_team_id=team.id,
                       preset_igl=team.igl, preset_carrier=team.carrier, preset_ai=team.ai)

    def with_confirmed_team(self):
        if not self.roster_ready:
            raise SeasonSaveError("チームの登録には5人の編成が必要です。")
        name = self.preset_name.strip()
        if any(team.name.casefold() == name.casefold() and team.id != self.editing_team_id for team in self.teams):
            raise SeasonSaveError("同じ名前の編成プリセットが登録されています。別のプリセット名を入力してください。")
        team = SeasonTeam(self.editing_team_id or uuid4().hex, name, self.roster,
                          self.preset_igl, self.preset_carrier, self.preset_ai)
        teams = tuple(team if old.id == team.id else old for old in self.teams)
        if self.editing_team_id is None:
            teams += (team,)
        candidate = replace(self, teams=teams, editing_team_id=team.id).with_registered_ratings()
        candidate.validate()
        return candidate

    def with_selected_team(self, team_id):
        candidate = replace(self, selected_team_id=team_id)
        candidate.validate()
        return candidate

    def with_roster(self, names):
        roster = tuple(names)
        candidate = replace(self, roster=roster,
                            preset_igl=self.preset_igl if self.preset_igl in roster else None,
                            preset_carrier=self.preset_carrier if self.preset_carrier in roster else None)
        candidate.validate()
        return candidate

    def with_preset_settings(self, *, igl=None, carrier=None, ai="default"):
        candidate = replace(self, preset_igl=igl, preset_carrier=carrier, preset_ai=ai)
        editing = self.team(self.editing_team_id)
        # An unchanged saved lineup can receive settings immediately. A new or
        # edited roster stays a draft until the five-player confirmation.
        if editing is not None and editing.roster == self.roster:
            candidate = replace(candidate, teams=tuple(
                replace(t, igl=igl, carrier=carrier, ai=ai) if t.id == editing.id else t for t in self.teams))
        candidate.validate()
        return candidate

    def with_team_name(self, name):
        candidate = replace(self, team_name=name.strip()).with_registered_ratings()
        candidate.validate()
        return candidate

    def with_preset_name(self, name):
        candidate = replace(self, preset_name=name.strip())
        candidate.validate()
        return candidate

    def with_added_players(self, names):
        players = list(self.owned_players)
        contracts = list(self.contracts)
        existing = {p.name for p in players}
        for name in names:
            owner = self.opponent_owner(name)
            if owner:
                raise SeasonSaveError(f"{name}は「{owner.name}」に所属しているため、所持選手に追加できません。")
            if name in existing:
                raise SeasonSaveError(f"{name}はすでに所持しています（または入力が重複しています）。")
            player = get_by_name(name)
            if player is None:
                raise SeasonSaveError(f"選手 {name} が見つかりません。既存のプレイヤー名を正確に入力してください。")
            player = self.salary_player(player)
            players.append(player)
            contracts = [c for c in contracts if c.player_name != name]
            contracts.append(initial_contract(player, self.game_month))
            existing.add(name)
        candidate = replace(self, owned_players=tuple(players), contracts=tuple(contracts))
        candidate.validate()
        return candidate

    def without_player(self, name):
        if name in self.roster:
            raise SeasonSaveError("先にその選手をロスターから外してください。")
        if self.player_in_saved_team(name):
            raise SeasonSaveError("先に登録済みチームの編成からその選手を外し、編成を確定してください。")
        candidate = replace(self, owned_players=tuple(p for p in self.owned_players if p.name != name),
                            contracts=tuple(replace(c, end_reason="released") if c.player_name == name else c for c in self.contracts))
        candidate.validate()
        return candidate

    def validate(self):
        if not isinstance(self.salary_mode, SalaryMode):
            raise SeasonSaveError("給与モードが不正です。")
        if not isinstance(self.salary_records, tuple):
            raise SeasonSaveError("給与一覧の形式が不正です。")
        if self.salary_mode == SalaryMode.STATIC:
            if self.salary_settings is not None or self.salary_records or self.salary_updated_month is not None:
                raise SeasonSaveError("静的月給モードの設定が不正です。")
        else:
            if not isinstance(self.salary_settings, SalarySettings) or not self.salary_records:
                raise SeasonSaveError("成績連動月給の設定または給与一覧がありません。")
            try:
                self.salary_settings.validate()
                names = set()
                for record in self.salary_records:
                    if not isinstance(record, SalaryRecord) or record.name in names:
                        raise SalaryDataError("給与一覧の選手が不正または重複しています。")
                    record.validate()
                    names.add(record.name)
            except SalaryDataError as exc:
                raise SeasonSaveError(str(exc)) from exc
            if type(self.salary_updated_month) is not int or self.salary_updated_month != self.game_month:
                raise SeasonSaveError("給与更新月とゲーム内の経過月数が一致しません。")
        if type(self.game_month) is not int or self.game_month < 0:
            raise SeasonSaveError("ゲーム内の経過月数が不正です。")
        if type(self.monthly_events_through) is not int or not 0 <= self.monthly_events_through <= self.game_month:
            raise SeasonSaveError("月次イベントの処理済み月が不正です。")
        event_ids = set()
        last_date = parse_date(self.start_date)
        if not isinstance(self.monthly_events, tuple):
            raise SeasonSaveError("月次イベント履歴の形式が不正です。")
        for event in self.monthly_events:
            if (not isinstance(event, MonthlyEvent) or not isinstance(event.id, str) or not event.id or event.id in event_ids
                    or not isinstance(event.kind, str) or event.kind not in {"recruitment", "recruitment_unfilled", "renewal", "departure", "month_completed"}
                    or event.team_id is not None and (not isinstance(event.team_id, str) or not event.team_id)
                    or not isinstance(event.team_name, str) or not event.team_name
                    or event.player_name is not None and (not isinstance(event.player_name, str) or not event.player_name)
                    or not isinstance(event.message, str) or not event.message):
                raise SeasonSaveError("月次イベント履歴が不正です。")
            try:
                date = parse_date(event.date)
            except CompetitionError as exc:
                raise SeasonSaveError("月次イベントの日付が不正です。") from exc
            if date < last_date or date > self.date or not 1 <= month_index(parse_date(self.start_date), date) <= self.monthly_events_through:
                raise SeasonSaveError("月次イベントの日付と処理済み月が一致しません。")
            event_ids.add(event.id)
            last_date = date
        if type(self.sponsor_active) is not bool:
            raise SeasonSaveError("スポンサー契約の状態が不正です。")
        if not isinstance(self.transferred_players, tuple) or any(not isinstance(n, str) or not n for n in self.transferred_players) or len(set(self.transferred_players)) != len(self.transferred_players):
            raise SeasonSaveError("移籍履歴が不正です。")
        if not isinstance(self.rated_results, tuple) or any(not isinstance(n, str) or not n for n in self.rated_results) or len(set(self.rated_results)) != len(self.rated_results):
            raise SeasonSaveError("レート更新済み試合の記録が不正です。")
        rating_ids = set()
        for rating in self.ratings:
            if (not isinstance(rating, SeasonRating) or not isinstance(rating.team_id, str) or not rating.team_id
                    or rating.team_id in rating_ids or not isinstance(rating.team_name, str) or not rating.team_name.strip()
                    or type(rating.value) not in (int, float) or not math.isfinite(rating.value) or rating.value < 0):
                raise SeasonSaveError("レーティング一覧が不正です。")
            rating_ids.add(rating.team_id)
        if type(self.money) is not int:
            raise SeasonSaveError("所持金は円単位の整数で保存してください。")
        if type(self.game_month) is not int or self.game_month < 0:
            raise SeasonSaveError("ゲーム内の経過月数が不正です。")
        try:
            if self.date < parse_date(self.start_date) or month_index(parse_date(self.start_date), self.date) != self.game_month:
                raise CompetitionError("ゲーム内の日付と経過月数が一致しません。")
            validate_periods(self.in_season_periods)
            validate_definitions(self.tournament_definitions)
        except CompetitionError as exc:
            raise SeasonSaveError(str(exc)) from exc
        if type(self.starter_selection_pending) is not bool:
            raise SeasonSaveError("初期キャラの選択状態が不正です。")
        pool_names = []
        for player in self.starter_candidates:
            validate_player(player)
            pool_names.append(player.name)
            if self.starter_selection_pending:
                owner = self.opponent_owner(player.name)
                if owner is not None:
                    raise SeasonSaveError(f"{player.name}は「{owner.name}」に所属しているため、初期キャラ候補にできません。")
        if len(set(pool_names)) != len(pool_names):
            raise SeasonSaveError("初期キャラ候補が重複しています。")
        if (any(not isinstance(name, str) for name in self.starter_selection)
                or len(self.starter_selection) > ROSTER_SIZE or len(set(self.starter_selection)) != len(self.starter_selection)
                or not set(self.starter_selection).issubset(pool_names)):
            raise SeasonSaveError("初期キャラの選択が不正です。候補から異なる5人まで選んでください。")
        if self.starter_selection_pending:
            if len(pool_names) < ROSTER_SIZE:
                raise SeasonSaveError("INITIAL_OWNED_PLAYERS に初期キャラ候補を5人以上設定してください。")
            if (self.owned_players or self.roster or self.teams or self.contracts or self.tournaments
                    or self.game_month != 0 or self.date != parse_date(self.start_date) or self.money != INITIAL_MONEY):
                raise SeasonSaveError("初期キャラの5人を確定してからゲームを開始してください。")
        elif self.starter_candidates and len(self.starter_selection) != ROSTER_SIZE:
            raise SeasonSaveError("入手済みの初期キャラの選択記録には5人が必要です。")
        if not isinstance(self.team_name, str) or not self.team_name.strip():
            raise SeasonSaveError("チーム名を入力してください。")
        if len(self.team_name) > 40:
            raise SeasonSaveError("チーム名は40文字以内にしてください。")
        if not isinstance(self.club_id, str) or not self.club_id:
            raise SeasonSaveError("自チームのIDが不正です。")
        if not isinstance(self.preset_name, str) or not self.preset_name.strip() or len(self.preset_name) > 40:
            raise SeasonSaveError("編成プリセット名は1～40文字で入力してください。")
        names = []
        for player in self.owned_players:
            validate_player(player)
            names.append(player.name)
        if len(set(names)) != len(names):
            raise SeasonSaveError("所持選手が重複しています。")
        contract_names = set()
        for contract in self.contracts:
            if not isinstance(contract, PlayerContract) or not isinstance(contract.player_name, str) or not contract.player_name:
                raise SeasonSaveError("選手の契約データが不正です。")
            if contract.player_name in contract_names:
                raise SeasonSaveError("選手の契約が重複しています。")
            contract_names.add(contract.player_name)
            if (type(contract.start_month) is not int or not 0 <= contract.start_month <= self.game_month
                    or type(contract.duration_months) is not int or type(contract.monthly_salary) is not int
                    or contract.monthly_salary < 0):
                raise SeasonSaveError("契約期間または月給が不正です。")
            if (not isinstance(contract.kind, str) or contract.kind not in CONTRACT_OPTIONS.values()
                    or (contract.kind == "short" and not 1 <= contract.duration_months <= 6)
                    or (contract.kind != "short" and contract.duration_months != {"year1": 12, "year2": 24, "year3": 36}.get(contract.kind))):
                raise SeasonSaveError("契約の種類と期間が一致しません。")
            if type(contract.team_loyalty) not in (int, float) or not math.isfinite(contract.team_loyalty):
                raise SeasonSaveError("チームへの忠誠が不正です。")
            if contract.end_reason not in (None, "left", "released"):
                raise SeasonSaveError("契約の終了理由が不正です。")
            if (contract.end_reason is None) != (contract.player_name in names):
                raise SeasonSaveError("契約と所持選手の所属が一致しません。")
        if not set(names).issubset(contract_names):
            raise SeasonSaveError("所持選手の契約が見つかりません。")
        if any(not isinstance(name, str) for name in self.roster):
            raise SeasonSaveError("ロスターの選手名が不正です。")
        if len(self.roster) > ROSTER_SIZE:
            raise SeasonSaveError("ロスターに登録できるのは5人までです。")
        if len(set(self.roster)) != len(self.roster):
            raise SeasonSaveError("同じ選手をロスターに重複登録できません。")
        if not set(self.roster).issubset(names):
            raise SeasonSaveError("所持していない選手はロスターに登録できません。")
        validate_preset_settings(self.roster, self.preset_igl, self.preset_carrier, self.preset_ai)
        team_ids = set()
        team_names = set()
        affiliations = {}
        for team in self.teams:
            if not isinstance(team, SeasonTeam) or not isinstance(team.id, str) or not team.id:
                raise SeasonSaveError("登録済みチームのIDが不正です。")
            if not isinstance(team.name, str) or not team.name.strip() or len(team.name) > 40:
                raise SeasonSaveError("登録済みチームの名前が不正です。")
            if team.id in team_ids or team.name.casefold() in team_names:
                raise SeasonSaveError("登録済みチームが重複しています。")
            if len(team.roster) != ROSTER_SIZE or any(not isinstance(name, str) for name in team.roster):
                raise SeasonSaveError("登録済みチームは5人の編成が必要です。")
            if len(set(team.roster)) != ROSTER_SIZE or not set(team.roster).issubset(names):
                raise SeasonSaveError("登録済みチームに重複または未所持の選手が含まれています。")
            if team.id == self.club_id:
                raise SeasonSaveError("編成プリセットと自チームのIDが重複しています。")
            validate_preset_settings(team.roster, team.igl, team.carrier, team.ai)
            team_ids.add(team.id)
            team_names.add(team.name.casefold())

        for team_id in (self.editing_team_id, self.selected_team_id):
            if team_id is not None and (not isinstance(team_id, str) or team_id not in team_ids):
                raise SeasonSaveError("編集・使用チームの指定が不正です。")
        team_names = {self.team_name.casefold()}
        team_ids.add(self.club_id)
        owned_names = set(names)
        for team in self.opponent_teams:
            if not isinstance(team, SeasonClub) or not isinstance(team.id, str) or not team.id:
                raise SeasonSaveError("シーズン所属チームのIDが不正です。")
            if not isinstance(team.name, str) or not team.name.strip() or len(team.name) > 40:
                raise SeasonSaveError("シーズン所属チーム名は1～40文字で設定してください。")
            if team.id in team_ids or team.name.casefold() in team_names:
                raise SeasonSaveError(f"シーズン所属チーム「{team.name}」が重複しています。")
            if type(team.transfer_multiplier) not in (int, float) or not math.isfinite(team.transfer_multiplier) or team.transfer_multiplier < 0:
                raise SeasonSaveError(f"「{team.name}」の移籍金倍率は0以上の有限の数値で設定してください。")
            if (not isinstance(team.contracts, tuple) or not isinstance(team.preferred_roles, tuple) or len(team.preferred_roles) > ROSTER_SIZE
                    or any(not isinstance(role, str) or not role for role in team.preferred_roles)):
                raise SeasonSaveError(f"「{team.name}」のロール構成が不正です。")
            contract_names = set()
            for contract in team.contracts:
                if (not isinstance(contract, PlayerContract) or not isinstance(contract.player_name, str) or not contract.player_name
                        or contract.player_name in contract_names or type(contract.start_month) is not int
                        or not 0 <= contract.start_month <= self.game_month or type(contract.duration_months) is not int
                        or type(contract.monthly_salary) is not int or contract.monthly_salary < 0
                        or contract.kind not in CONTRACT_OPTIONS.values()
                        or contract.kind == "short" and not 1 <= contract.duration_months <= 6
                        or contract.kind != "short" and contract.duration_months != {"year1": 12, "year2": 24, "year3": 36}.get(contract.kind)
                        or type(contract.team_loyalty) not in (int, float) or not math.isfinite(contract.team_loyalty)
                        or contract.end_reason not in (None, "left", "released")
                        or (contract.end_reason is None) != (contract.player_name in team.members)):
                    raise SeasonSaveError(f"「{team.name}」の契約データが不正です。")
                contract_names.add(contract.player_name)
            for player in team.players:
                validate_player(player)
                name = player.name
                if name in owned_names:
                    raise SeasonSaveError(f"{name}は自分の所持選手と「{team.name}」に重複所属しています。")
                if name in affiliations:
                    raise SeasonSaveError(f"{name}が「{affiliations[name]}」と「{team.name}」に重複所属しています。")
                affiliations[name] = team.name
            for label, value in (("IGL", team.igl), ("キャリアー", team.carrier)):
                if value is not None and (not isinstance(value, str) or value not in team.roster):
                    raise SeasonSaveError(f"「{team.name}」の{label}は、先頭5人の出場選手から指定してください。")
            from roster_select import TEAM_AI_OPTIONS
            if not isinstance(team.ai, str) or team.ai not in TEAM_AI_OPTIONS.values():
                raise SeasonSaveError(f"「{team.name}」のAI設定が不正です: {team.ai!r}")
            team_ids.add(team.id)
            team_names.add(team.name.casefold())

        try:
            self._validate_competitions()
        except (CompetitionError, TypeError, AttributeError) as exc:
            raise SeasonSaveError(f"大会データが不正です: {exc}") from exc

    def _validate_competitions(self):
        from roster_select import TEAM_AI_OPTIONS
        seen = set()
        for run in self.tournaments:
            if not isinstance(run, TournamentProgress) or run.tournament_id in seen:
                raise CompetitionError("大会進行データが重複または不正です。")
            seen.add(run.tournament_id)
            event = self.tournament_definition(run.tournament_id)
            if event is None or type(run.completed) is not bool or type(run.declined) is not bool:
                raise CompetitionError("大会設定または進行状態が不正です。")
            if type(run.prize_paid) is not int or run.prize_paid < 0 or type(run.seed) is not int or not 0 <= run.seed < 2**31:
                raise CompetitionError("大会賞金または乱数シードが不正です。")
            if bool(run.results) != (run.last_match_date is not None):
                raise CompetitionError("大会の最終試合日が結果と一致しません。")
            if run.last_match_date is not None:
                if not parse_date(event.start_date) <= parse_date(run.last_match_date) <= self.date:
                    raise CompetitionError("大会の最終試合日が不正です。")
            if run.completed_date != (run.last_match_date if run.completed and not run.declined else None):
                raise CompetitionError("大会の終了日が完了状態と一致しません。")
            if run.declined:
                if not event.participation_optional or not event.allow_player_entry or not run.completed or run.entrants or run.results or run.ranking or run.prize_paid or run.own_team_id is not None:
                    raise CompetitionError("不参加の大会データが不正です。")
                continue
            if len(run.entrants) != event.team_count:
                raise CompetitionError("大会の参加チーム数が設定と一致しません。")
            ids, names, player_names = set(), set(), set()
            for team in run.entrants:
                if not isinstance(team, CompetitionTeam) or not isinstance(team.id, str) or not team.id or team.id in ids:
                    raise CompetitionError("大会の参加チームIDが重複または不正です。")
                if not isinstance(team.name, str) or not team.name.strip() or team.name.casefold() in names:
                    raise CompetitionError("大会の参加チーム名が重複または不正です。")
                ids.add(team.id)
                names.add(team.name.casefold())
                if len(team.players) != ROSTER_SIZE:
                    raise CompetitionError("大会には5人のロスターが必要です。")
                members = []
                for player in team.players:
                    validate_player(player)
                    if player.name in player_names:
                        raise CompetitionError("大会内で選手が重複所属しています。")
                    player_names.add(player.name)
                    members.append(player.name)
                if team.ai not in TEAM_AI_OPTIONS.values() or team.igl not in members or team.carrier not in members:
                    raise CompetitionError("大会チームのAI・IGL・キャリアーが不正です。")
            if (run.own_team_id is None) == event.allow_player_entry or (run.own_team_id is not None and run.own_team_id not in ids):
                raise CompetitionError("プレイヤーの参加状態が不正です。")
            if (run.own_team_id is not None and (run.own_team_id != self.club_id or not isinstance(run.preset_id, str) or not run.preset_id)
                    or run.own_team_id is None and run.preset_id is not None):
                raise CompetitionError("大会の自チーム・編成プリセットの指定が不正です。")
            pending, ranking = next_match(event, run)
            for score in run.results:
                if score.decided_by_rating and run.own_team_id in (score.left_id, score.right_id):
                    raise CompetitionError("レート判定は他チーム同士の試合にのみ使用できます。")
            if run.completed != (pending is None) or run.ranking != ranking:
                raise CompetitionError("大会の完了状態または順位が試合結果と一致しません。")
            expected = event.prizes.get(ranking.index(run.own_team_id) + 1, 0) if ranking and run.own_team_id else 0
            if run.prize_paid != expected:
                raise CompetitionError("大会順位と受け取り賞金が一致しません。")
            if not run.completed and run.own_team_id:
                team = self.team(run.preset_id)
                snapshot = next(t for t in run.entrants if t.id == run.own_team_id)
                departed = any(self.player(p.name) is None for p in snapshot.players)
                if (team is None and not departed) or (team is not None and team.roster != tuple(p.name for p in snapshot.players)):
                    raise CompetitionError("大会参加中のチームの編成は変更できません。大会終了後に変更してください。")


def configured_season_teams(existing_teams=(), transferred_players=(), *, game_month=0, monthly_events=()):
    from realtime_season_teams import SEASON_TEAMS
    if not isinstance(SEASON_TEAMS, (list, tuple)):
        raise SeasonSaveError("SEASON_TEAMS はチーム設定のリストにしてください。")
    saved_players = {p.name: p for team in existing_teams for p in team.players}
    saved_teams = {team.name: team for team in existing_teams}
    automatic_signings = {(event.team_id, event.player_name) for event in monthly_events if event.kind == "recruitment"}
    automatic_departures = {(event.team_id, event.player_name) for event in monthly_events if event.kind == "departure"}
    def excluded(team_name, player_name):
        old = saved_teams.get(team_name) if isinstance(team_name, str) else None
        return player_name in transferred_players or old is not None and (old.id, player_name) in automatic_departures and player_name not in old.members
    configured_names = {p for item in SEASON_TEAMS if isinstance(item, dict) and isinstance(item.get("players"), (list, tuple))
                        for p in item["players"] if isinstance(p, str) and not excluded(item.get("name"), p)}
    teams = []
    for item in SEASON_TEAMS:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise SeasonSaveError("各シーズンチームに name（チーム名）を設定してください。")
        name = item["name"].strip()
        saved_team = saved_teams.get(name)
        members = item.get("players")
        if not isinstance(members, (list, tuple)) or any(not isinstance(p, str) or not p.strip() for p in members):
            raise SeasonSaveError(f"「{name}」の players に所属選手名のリストを設定してください。")
        if len(members) < ROSTER_SIZE or len(set(members)) != len(members):
            raise SeasonSaveError(f"「{name}」には重複のない5人以上の所属選手が必要です。")
        players = []
        for member in members:
            if excluded(name, member):
                continue
            player = saved_players.get(member) or get_by_name(member)
            if player is None:
                raise SeasonSaveError(f"「{name}」の選手 {member} が見つかりません。")
            players.append(player)
        if saved_team:
            players.extend(p for p in saved_team.players if (saved_team.id, p.name) in automatic_signings
                           and p.name not in configured_names and p.name not in transferred_players)
        contracts = {c.player_name: c for c in saved_team.contracts} if saved_team else {}
        members_now = {p.name for p in players}
        contracts = {name: c if name in members_now else replace(c, end_reason=c.end_reason or "released")
                     for name, c in contracts.items()}
        for player in players:
            if player.name not in contracts or contracts[player.name].end_reason is not None:
                contracts[player.name] = initial_contract(player, game_month)
        roster = {p.name for p in players[:ROSTER_SIZE]}
        igl, carrier = item.get("igl"), item.get("carrier")
        if igl in transferred_players or igl in members[:ROSTER_SIZE] and igl not in roster:
            igl = None
        if carrier in transferred_players or carrier in members[:ROSTER_SIZE] and carrier not in roster:
            carrier = None
        teams.append(SeasonClub(
            "world-" + uuid5(NAMESPACE_URL, "realtime-season:" + name).hex,
            name, tuple(players), igl=igl, carrier=carrier, ai=item.get("ai", "default"),
            transfer_multiplier=item.get("transfer_multiplier", 12.0),
            preferred_roles=saved_team.preferred_roles if saved_team and saved_team.preferred_roles else tuple(p.role for p in players[:ROSTER_SIZE]),
            contracts=tuple(contracts.values()),
        ))
    # Check team names, membership, reserves, and ability snapshots as one batch.
    SeasonState("マイチーム", (), opponent_teams=tuple(teams), game_month=game_month).validate()
    return tuple(teams)


def new_season(starter_names=None, *, salary_mode=SalaryMode.STATIC):
    choose_starters = starter_names is None
    if starter_names is None:
        from realtime_season_config import INITIAL_OWNED_PLAYERS
        starter_names = INITIAL_OWNED_PLAYERS
    if not isinstance(starter_names, (list, tuple)) or any(
        not isinstance(name, str) or not name.strip() for name in starter_names
    ):
        raise SeasonSaveError("初期所持プレイヤーは、プレイヤー名の文字列を並べたリストで設定してください。")
    try:
        start, periods, events = configured_calendar()
    except CompetitionError as exc:
        raise SeasonSaveError(str(exc)) from exc
    state = SeasonState("マイチーム", (), opponent_teams=configured_season_teams(), start_date=start, game_date=start,
                        in_season_periods=periods, tournament_definitions=events).with_registered_ratings()
    if not choose_starters:
        # Explicit names remain available for administrative setup and simulations.
        state = state.with_added_players(starter_names)
        if SalaryMode(salary_mode) == SalaryMode.KD_DYNAMIC:
            state = replace(state, salary_mode=SalaryMode.KD_DYNAMIC, salary_settings=configured_salary_settings()).with_updated_salaries(initial=True)
        return state
    candidates = []
    for name in starter_names:
        player = get_by_name(name)
        if player is None:
            raise SeasonSaveError(f"初期キャラ候補 {name} が見つかりません。")
        candidates.append(player)
    state = replace(state, starter_candidates=tuple(candidates), starter_selection_pending=True)
    state.validate()
    return state.with_salary_mode(salary_mode)


class SeasonStore:
    def __init__(self, path=DEFAULT_SAVE_PATH):
        self.path = Path(path).resolve()

    def import_season_teams(self, state):
        candidate = state.with_opponent_teams(configured_season_teams(state.opponent_teams, state.transferred_players,
                                            game_month=state.game_month, monthly_events=state.monthly_events))
        self.save(candidate)
        return candidate

    def import_competitions(self, state):
        try:
            _, periods, definitions = configured_calendar()
            protected = {run.tournament_id for run in state.tournaments}
            # Started and completed events retain their rules and prize snapshots.
            events = tuple(event for event in state.tournament_definitions if event.id in protected)
            events += tuple(event for event in definitions if event.id not in protected)
            candidate = replace(state, in_season_periods=periods, tournament_definitions=events)
            self.save(candidate)
            return candidate
        except CompetitionError as exc:
            raise SeasonSaveError(str(exc)) from exc

    def load_or_create(self):
        # Only an absent save starts a new game. Unreadable saves are preserved.
        try:
            contents = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            state = new_season()
            self.save(state)
            return state
        except UnicodeError as exc:
            raise SeasonSaveError("セーブデータはUTF-8形式で保存してください。") from exc
        try:
            data = json.loads(contents)
            if not isinstance(data, dict):
                raise SeasonSaveError("セーブデータはJSONオブジェクトである必要があります。")
            if type(data.get("version")) is not int or data["version"] not in range(1, SAVE_VERSION + 1):
                raise SeasonSaveError("このセーブデータのバージョンには対応していません。")
            def load_player(row):
                return player_from_save(row, legacy_loyalty=data["version"] < 7)
            if not isinstance(data.get("owned_players"), list) or not isinstance(data.get("roster"), list):
                raise SeasonSaveError("所持選手またはロスターの形式が不正です。")
            teams = ()
            editing_team_id = None
            selected_team_id = None
            if data["version"] == 1 and len(data["roster"]) == ROSTER_SIZE:
                teams = (SeasonTeam("legacy-team", data["team_name"], tuple(data["roster"])),)
                editing_team_id = "legacy-team"
            elif data["version"] >= 2:
                if not isinstance(data.get("teams"), list):
                    raise SeasonSaveError("登録済みチームの形式が不正です。")
                parsed_teams = []
                for item in data["teams"]:
                    if not isinstance(item, dict) or not isinstance(item.get("roster"), list):
                        raise SeasonSaveError("登録済みチームの形式が不正です。")
                    parsed_teams.append(SeasonTeam(item["id"], item["name"], tuple(item["roster"]),
                        item.get("igl"), item.get("carrier"), item.get("ai", "default")))
                teams = tuple(parsed_teams)
                editing_team_id = data.get("editing_team_id")
                selected_team_id = data.get("selected_team_id")
            opponent_teams = ()
            if data["version"] >= 3:
                if not isinstance(data.get("opponent_teams"), list):
                    raise SeasonSaveError("シーズン所属チームの形式が不正です。")
                clubs = []
                for item in data["opponent_teams"]:
                    if not isinstance(item, dict) or not isinstance(item.get("players"), list):
                        raise SeasonSaveError("シーズン所属チームの形式が不正です。")
                    if data["version"] >= 9 and (not isinstance(item.get("contracts"), list) or not isinstance(item.get("preferred_roles"), list)):
                        raise SeasonSaveError("他チームの契約またはロール構成の形式が不正です。")
                    clubs.append(SeasonClub(
                        item["id"], item["name"], tuple(load_player(p) for p in item["players"]),
                        igl=item.get("igl"), carrier=item.get("carrier"), ai=item.get("ai", "default"),
                        transfer_multiplier=item.get("transfer_multiplier", 12.0),
                        preferred_roles=tuple(item.get("preferred_roles", ())),
                        contracts=tuple(PlayerContract(**c) for c in item.get("contracts", ())),
                    ))
                opponent_teams = tuple(clubs)
            owned_players = tuple(load_player(p) for p in data["owned_players"])
            if data["version"] >= 4:
                if not isinstance(data.get("contracts"), list):
                    raise SeasonSaveError("契約データの形式が不正です。")
                contracts = tuple(PlayerContract(**row) for row in data["contracts"])
                money = data["money"]
                game_month = data["game_month"]
            else:
                contracts = tuple(initial_contract(p) for p in owned_players)
                money = INITIAL_MONEY
                game_month = 0
            if data["version"] >= 5:
                start_date = data["start_date"]
                game_date = data["game_date"]
                parse_date(game_date)
                periods = data["in_season_periods"]
                validate_periods(periods)
                periods = tuple(tuple(pair) for pair in periods)
                if not isinstance(data.get("tournament_definitions"), list) or not isinstance(data.get("tournaments"), list):
                    raise SeasonSaveError("大会設定または進行データの形式が不正です。")
                events = tuple(definition_from_dict(row) for row in data["tournament_definitions"])
                runs = []
                for row in data["tournaments"]:
                    if not isinstance(row, dict):
                        raise SeasonSaveError("大会進行データの形式が不正です。")
                    values = dict(row)
                    entrants = []
                    for team in values["entrants"]:
                        team = dict(team)
                        team["players"] = tuple(load_player(p) for p in team["players"])
                        entrants.append(CompetitionTeam(**team))
                    values["entrants"] = tuple(entrants)
                    values["results"] = tuple(SeriesScore(**result) for result in values["results"])
                    values["ranking"] = tuple(values["ranking"])
                    if data["version"] < 10:
                        # Older saves have no match dates. Keep every result and
                        # resume on the next day without rewriting the saved file.
                        values.setdefault("last_match_date", game_date if values["results"] else None)
                        values.setdefault("completed_date", values["last_match_date"] if values["completed"] and not values["declined"] else None)
                    runs.append(TournamentProgress(**values))
                runs = tuple(runs)
            else:
                start_date, periods, events = configured_calendar()
                game_date = add_months(parse_date(start_date), game_month).isoformat()
                runs = ()
            if data["version"] >= 6:
                if not isinstance(data.get("starter_candidates"), list) or not isinstance(data.get("starter_selection"), list):
                    raise SeasonSaveError("初期キャラ候補または選択の形式が不正です。")
                starter_candidates = tuple(load_player(p) for p in data["starter_candidates"])
                starter_selection = tuple(data["starter_selection"])
                starter_selection_pending = data["starter_selection_pending"]
            else:
                # Existing saves keep their inventory; never discard players to restart selection.
                starter_candidates, starter_selection, starter_selection_pending = (), (), False
            ratings, rated_results, sponsor_active, transferred_players = (), (), True, ()
            if data["version"] >= 8:
                if any(not isinstance(data.get(key), list) for key in ("ratings", "rated_results", "transferred_players")):
                    raise SeasonSaveError("レーティングまたは移籍履歴の形式が不正です。")
                ratings = tuple(SeasonRating(**row) for row in data["ratings"])
                rated_results = tuple(data["rated_results"])
                sponsor_active = data["sponsor_active"]
                transferred_players = tuple(data["transferred_players"])
            monthly_events, monthly_events_through = (), game_month
            if data["version"] >= 9:
                if not isinstance(data.get("monthly_events"), list):
                    raise SeasonSaveError("月次イベント履歴の形式が不正です。")
                monthly_events = tuple(MonthlyEvent(**row) for row in data["monthly_events"])
                monthly_events_through = data["monthly_events_through"]
            opponent_teams = tuple(replace(club,
                preferred_roles=club.preferred_roles or tuple(p.role for p in club.players[:ROSTER_SIZE]),
                contracts=club.contracts or tuple(initial_contract(p, game_month) for p in club.players)) for club in opponent_teams)
            salary_mode, salary_settings, salary_rows, salary_month = SalaryMode.STATIC, None, (), None
            if data["version"] >= 14:
                salary_mode = SalaryMode(data["salary_mode"])
                settings = data["salary_settings"]
                if settings is not None:
                    settings = dict(settings)
                    settings["fixed_salaries"] = tuple(tuple(row) for row in settings["fixed_salaries"])
                    settings["excluded_files"] = tuple(settings.get("excluded_files", ()))
                    salary_settings = SalarySettings(**settings)
                if not isinstance(data["salary_records"], list):
                    raise SeasonSaveError("給与一覧の形式が不正です。")
                salary_rows = tuple(SalaryRecord(**row) for row in data["salary_records"])
                salary_month = data["salary_updated_month"]
            state = SeasonState(
                team_name=data["team_name"],
                owned_players=owned_players,
                roster=tuple(data["roster"]),
                teams=teams,
                editing_team_id=editing_team_id,
                selected_team_id=selected_team_id,
                opponent_teams=opponent_teams,
                money=money,
                game_month=game_month,
                contracts=contracts,
                start_date=start_date,
                game_date=game_date,
                in_season_periods=periods,
                tournament_definitions=events,
                tournaments=runs,
                starter_candidates=starter_candidates,
                starter_selection=starter_selection,
                starter_selection_pending=starter_selection_pending,
                ratings=ratings,
                rated_results=rated_results,
                sponsor_active=sponsor_active,
                transferred_players=transferred_players,
                monthly_events=monthly_events,
                monthly_events_through=monthly_events_through,
                preset_name=data["preset_name"] if data["version"] >= 11 else data.get("preset_name", data["team_name"]),
                club_id=data["club_id"] if data["version"] >= 11 else data.get("club_id", PLAYER_CLUB_ID),
                preset_igl=data.get("preset_igl"),
                preset_carrier=data.get("preset_carrier"),
                preset_ai=data.get("preset_ai", "default"),
                salary_mode=salary_mode, salary_settings=salary_settings,
                salary_records=salary_rows, salary_updated_month=salary_month,
            )
            old_layout = data["version"] < 11 and "club_id" not in data
            required_ids = {t.id for t in state.opponent_teams}
            if old_layout:
                required_ids.update(t.id for t in state.teams)
            else:
                required_ids.add(state.club_id)
            if data["version"] >= 8 and not required_ids.issubset({r.team_id for r in state.ratings}):
                raise SeasonSaveError("所属チームのレーティングがセーブデータにありません。")
            if old_layout:
                old_ids = {t.id for t in state.teams} | {r.own_team_id for r in state.tournaments if r.own_team_id}
                active = next((r for r in state.tournaments if not r.completed and r.own_team_id), None)
                source_id = active.own_team_id if active else state.selected_team_id or (state.teams[0].id if state.teams else None)
                source = state.team(source_id)
                source = source or next((t for r in state.tournaments for t in r.entrants if t.id == source_id), None)
                actual_name = source.name if source else state.team_name
                own_rating = next((r.value for r in state.ratings if r.team_id == source_id), DEFAULT_TEAM_RATING)
                migrated_runs = []
                def club_key(key):
                    return state.club_id if key in old_ids else key
                for run in state.tournaments:
                    migrated_runs.append(replace(run,
                        own_team_id=club_key(run.own_team_id) if run.own_team_id else None,
                        preset_id=run.own_team_id,
                        entrants=tuple(replace(t, id=club_key(t.id), name=actual_name) if t.id in old_ids else t for t in run.entrants),
                        results=tuple(replace(s, left_id=club_key(s.left_id), right_id=club_key(s.right_id)) for s in run.results),
                        ranking=tuple(club_key(key) for key in run.ranking)))
                state = replace(state, team_name=actual_name, tournaments=tuple(migrated_runs),
                    ratings=(*tuple(r for r in state.ratings if r.team_id not in old_ids and r.team_id != state.club_id),
                             SeasonRating(state.club_id, actual_name, own_rating)))
            state = state.with_registered_ratings()
            state.validate()
            return state
        except (KeyError, TypeError, ValueError) as exc:
            raise SeasonSaveError(f"セーブデータを読み込めません: {exc}") from exc

    def save(self, state):
        state.validate()
        data = {
            "version": SAVE_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "team_name": state.team_name,
            "preset_name": state.preset_name,
            "preset_igl": state.preset_igl,
            "preset_carrier": state.preset_carrier,
            "preset_ai": state.preset_ai,
            "club_id": state.club_id,
            "salary_mode": state.salary_mode.value,
            "salary_settings": asdict(state.salary_settings) if state.salary_settings is not None else None,
            "salary_records": [asdict(r) for r in state.salary_records],
            "salary_updated_month": state.salary_updated_month,
            # Keep individual ability snapshots so future growth can be saved.
            "owned_players": [asdict(player) for player in state.owned_players],
            "roster": list(state.roster),
            "teams": [asdict(team) for team in state.teams],
            "editing_team_id": state.editing_team_id,
            "selected_team_id": state.selected_team_id,
            "opponent_teams": [asdict(team) for team in state.opponent_teams],
            "money": state.money,
            "game_month": state.game_month,
            "contracts": [asdict(contract) for contract in state.contracts],
            "start_date": state.start_date,
            "game_date": state.date.isoformat(),
            "in_season_periods": state.in_season_periods,
            "tournament_definitions": [asdict(event) for event in state.tournament_definitions],
            "tournaments": [asdict(run) for run in state.tournaments],
            "starter_candidates": [asdict(p) for p in state.starter_candidates],
            "starter_selection": list(state.starter_selection),
            "starter_selection_pending": state.starter_selection_pending,
            "ratings": [asdict(rating) for rating in state.ratings],
            "rated_results": list(state.rated_results),
            "sponsor_active": state.sponsor_active,
            "transferred_players": list(state.transferred_players),
            "monthly_events": [asdict(event) for event in state.monthly_events],
            "monthly_events_through": state.monthly_events_through,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
