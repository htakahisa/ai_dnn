"""Persistent player inventory and lineup for the real-time season mode."""

from dataclasses import asdict, dataclass, field, fields, replace
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
from realtime_season_config import DEFAULT_TEAM_AI
from season_ratings import (
    DEFAULT_TEAM_RATING,
    SeasonRating,
    expected_score,
    series_ratings,
)
from season_monthly_events import MonthlyEvent, process_monthly_events
from season_transfers import (
    TransferOffer,
    resolve_club_memberships,
    with_randomized_clubs,
)
from season_world_levels import (
    WorldLevel,
    WorldLevelError,
    scale_enemy_player,
    validate_world_level,
    world_level_for_rank,
)
from season_training import MAX_TRAINING_LEVEL, training_terms
from season_scouting import remaining as scout_remaining, validate_uses
from season_loyalty import (
    loyalty_for, remember_loyalties, validate_loyalties, validate_contract_memory,
    signing_loyalty, contract_refused, CONTRACT_LOYALTY, BENCHED_LOYALTY_LOSS,
)
from season_history import cash_item, export_history, record_state, validate_history
from functools import cached_property
import realtime_season_pair_familiarity as pair_settings
from season_pair_familiarity import (
    advance_pair_days,
    decode_pair_days,
    encode_pair_days,
    initial_pair_days,
    memberships,
    team_metrics,
    validate_pair_days,
    validate_settings as validate_pair_settings,
    PairNews,
    add_separation_news,
    daily_news,
    validate_news,
)
import realtime_season_rival_economy as rival_settings
from season_rival_economy import (
    can_sign,
    monthly_settlement,
    offer_terms,
    signed_contract,
)
from season_salary import (
    SalaryMode,
    SalaryRecord,
    SalarySettings,
    SalaryDataError,
    configured_salary_settings,
    salary_records,
    log_salary_change,
    clamp_change,
)
from season_competitions import (
    CompetitionError,
    CompetitionTeam,
    SeriesScore,
    TournamentProgress,
    add_months,
    configured_calendar,
    definition_from_dict,
    extend_annual_calendar,
    month_index,
    next_match,
    parse_date,
    phase_for,
    player_eliminated,
    validate_definitions,
    validate_periods,
)

FORCED_OFFER_LOYALTY = 10
SAVE_VERSION = 28
PLAYER_CLUB_ID = "player_club"
ROSTER_SIZE = 5
INITIAL_CANDIDATE_COUNT = 7
INITIAL_MONEY = 1_000_000
EXPIRED_ROSTER_WARNING = "このプレイヤーは契約が終了しているため編成できません"
CONTRACT_OPTIONS = {
    "短期契約": "short",
    "365日契約": "year1",
    "730日契約": "year2",
    "1095日契約": "year3",
}
DEFAULT_SAVE_PATH = (
    Path(__file__).resolve().parent / "data" / "realtime_season" / "save.json"
)


class SeasonSaveError(ValueError):
    """Invalid or unsupported season data; never reset it implicitly."""


def validate_initial_rating(value, team_name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise SeasonSaveError(
            f"「{team_name}」の初期レートは0以上の有限の数値で設定してください。"
        )


@dataclass(frozen=True)
class ContractTerms:
    kind: str
    # Wage-budget multiples; contract expiration uses days exclusively.
    months: int
    monthly_salary: int

    @property
    def days(self):
        return contract_duration_days(self.kind, self.months)

    @property
    def required_funds(self):
        return self.monthly_salary * self.months

    @property
    def signing_bonus(self):
        return self.monthly_salary * 3

    @property
    def total_required_funds(self):
        return self.signing_bonus + self.required_funds


def contract_duration_days(kind, months):
    return months * 30 if kind == "short" else {"year1": 365, "year2": 730, "year3": 1095}[kind]


def contract_terms(player, kind, short_months=6):
    validate_player(player)
    if kind == "short":
        if type(short_months) is not int or not 1 <= short_months <= 6:
            raise SeasonSaveError("短期契約の期間は30～180日（30日単位）から選択してください。")
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
    # Retain monthly metadata for old saves and wage-budget calculations.
    start_month: int
    duration_months: int
    team_loyalty: float = 50.0
    end_reason: str | None = None
    signing_number: int = 1
    expired_on: str | None = None
    signed_on: str | None = None

    @property
    def duration_days(self):
        return contract_duration_days(self.kind, self.duration_months)

    def starts_on(self, season_start):
        # Old saves only recorded the signing month. Use its first day.
        return (parse_date(self.signed_on) if self.signed_on is not None
                else add_months(parse_date(season_start), self.start_month))

    def ends_on(self, season_start):
        return self.starts_on(season_start) + timedelta(days=self.duration_days)

    @property
    def end_month(self):
        return self.start_month + self.duration_months

    def active(self, day, season_start):
        return self.end_reason is None and self.starts_on(season_start) <= day < self.ends_on(season_start)

    def elapsed(self, day, season_start):
        return max(0, min((day - self.starts_on(season_start)).days, self.duration_days))

    def remaining(self, day, season_start):
        return max(0, (self.ends_on(season_start) - day).days) if self.end_reason is None else 0


def initial_contract(player, month=0, *, signed_on=None):
    if player.loyalty == 0:
        return PlayerContract(player.name, "short", player.monthly_salary, month, 6, CONTRACT_LOYALTY["short"][0], signed_on=signed_on)
    return PlayerContract(player.name, "year1", player.monthly_salary, month, 12, CONTRACT_LOYALTY["year1"][0], signed_on=signed_on)


def player_from_save(row, *, legacy_loyalty=False):
    """Add catalog economy settings to old snapshots without resetting abilities."""
    if not isinstance(row, dict):
        raise SeasonSaveError("選手のデータ形式が不正です。")
    row = dict(row)
    # Only saved values use the old scale; missing values come from today's catalog.
    if legacy_loyalty and "loyalty" in row:
        value = row["loyalty"]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
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
    ai: str = DEFAULT_TEAM_AI


@dataclass(frozen=True)
class SeasonClub:
    id: str
    name: str
    players: tuple[CharacterStats, ...]
    igl: str | None = None
    carrier: str | None = None
    ai: str = DEFAULT_TEAM_AI
    transfer_multiplier: float = 12.0
    preferred_roles: tuple[str, ...] = ()
    contracts: tuple[PlayerContract, ...] = ()
    regular_members: tuple[str, ...] = ()
    regular_igl: str | None = None
    regular_carrier: str | None = None
    money: int = rival_settings.INITIAL_MONEY
    sponsor_active: bool = True
    acquired_members: tuple[str, ...] = ()
    world_level_lock: WorldLevel | None = None
    initial_rating: float = DEFAULT_TEAM_RATING
    # Shared configured players keep their chosen initial club during mixing.
    initial_shared_members: tuple[str, ...] = ()

    @property
    def members(self):
        return tuple(player.name for player in self.players)

    @property
    def roster(self):
        return self.members[:ROSTER_SIZE]

    @property
    def effective_igl(self):
        return (
            self.igl
            if self.igl is not None
            else (
                max(self.players[:ROSTER_SIZE], key=lambda p: p.iq).name
                if self.players
                else None
            )
        )

    @property
    def effective_carrier(self):
        return (
            self.carrier
            if self.carrier is not None
            else (self.players[0].name if self.players else None)
        )


def validate_preset_settings(roster, igl, carrier, ai):
    from roster_select import TEAM_AI_OPTIONS

    for label, name in (("IGL", igl), ("キャリアー", carrier)):
        if name is not None and (not isinstance(name, str) or name not in roster):
            raise SeasonSaveError(
                f"{label}はこのプリセットのロスターから選択してください。"
            )
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
            raise SeasonSaveError(
                f"{player.name}の月給は円単位の整数で設定してください。"
            )
        if key in {"research_level", "aim_lab_level"} and (
            type(value) is not int or value > MAX_TRAINING_LEVEL
        ):
            raise SeasonSaveError(
                f"{player.name}の育成レベルは0～{MAX_TRAINING_LEVEL}の整数にしてください。"
            )
        maximum = (
            1
            if key in {"hs_pct", "dodge_pct"}
            else (10 if key in {"form_variance", "mental", "loyalty"} else None)
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
    transfer_offers: tuple[TransferOffer, ...] = ()
    preset_name: str = "編成1"
    club_id: str = PLAYER_CLUB_ID
    preset_igl: str | None = None
    preset_carrier: str | None = None
    preset_ai: str = DEFAULT_TEAM_AI
    salary_mode: SalaryMode = SalaryMode.STATIC
    salary_settings: SalarySettings | None = None
    salary_records: tuple[SalaryRecord, ...] = ()
    salary_updated_month: int | None = None
    developed_players: tuple[CharacterStats, ...] = ()
    world_level_lock: WorldLevel | None = None
    # Audit metadata does not change equality of gameplay states.
    history: tuple[dict, ...] = field(default=(), compare=False, repr=False)
    pair_days: dict[tuple[str, str], int] = field(default_factory=dict)
    pair_familiarity_seed: int = field(
        default_factory=lambda: pair_settings.INITIAL_SEED
    )
    pair_news: tuple[PairNews, ...] = ()
    # Suppress intermediate month-end notices until the day's final rosters exist.
    pair_news_deferred: bool = field(default=False, compare=False, repr=False)
    scout_uses: tuple[tuple[str, str], ...] = ()
    day_advance_pending: bool = False
    team_loyalties: dict[str, dict[str, float]] = field(default_factory=dict)
    contract_signings: dict[str, dict[str, int]] = field(default_factory=dict)
    contract_bans: tuple[tuple[str, str], ...] = ()

    def team_loyalty(self, name, team_id=None):
        return loyalty_for(self, name, self.club_id if team_id is None else team_id)

    def contract_refused(self, name, team_id=None):
        return contract_refused(self, name, self.club_id if team_id is None else team_id)

    def scout_remaining(self, team_id=None):
        return scout_remaining(self, self.club_id if team_id is None else team_id)

    @property
    def scout_allowance_text(self):
        count = self.scout_remaining()
        return (
            "無制限（オフシーズン）"
            if count is None
            else f"残り{count}回（インシーズン）"
        )

    @property
    def day_action_blocked(self):
        if self.day_advance_pending:
            return "完了したアクションの1日進行を待っています。大会への参加登録後、日付を進めてください。"
        return (
            "出場中の大会があるため、スカウト・研究・エイムラボ・スクリムはできません。大会を進めてください。"
            if any(run.own_team_id == self.club_id for run in self.active_tournaments)
            else ""
        )

    def check_day_action(self):
        if self.day_action_blocked:
            raise SeasonSaveError(self.day_action_blocked)

    def scout_blocked(self, team_id=None):
        if team_id is None or team_id == self.club_id:
            if self.day_action_blocked:
                return self.day_action_blocked
        return (
            "インシーズンのスカウト上限に達しています。次の期間まで獲得できません。"
            if self.scout_remaining(team_id) == 0
            else ""
        )

    def _with_scout_use(self, team_id):
        return replace(
            self, scout_uses=(*self.scout_uses, (self.date.isoformat(), team_id))
        )

    @cached_property
    def pair_ranking(self):
        return tuple(
            sorted(self.pair_days.items(), key=lambda item: (-item[1], item[0]))
        )

    @cached_property
    def pair_partners(self):
        partners = {}
        for (a, b), days in self.pair_ranking:
            partners.setdefault(a, []).append((b, days))
            partners.setdefault(b, []).append((a, days))
        return partners

    def pair_metrics(self, starters):
        return team_metrics(self.pair_days, starters)

    def pair_starters(self, team_id):
        if team_id == self.club_id:
            preset = self.selected_team or (self.teams[0] if self.teams else None)
            return preset.roster if preset else self.roster
        club = next((c for c in self.opponent_teams if c.id == team_id), None)
        return club.roster if club else ()

    def _record_history(self, kind, *, income=(), expenses=()):
        candidate = remember_loyalties(self)
        if self.history and not self.pair_news_deferred:
            candidate = add_separation_news(candidate, self.history[-1]["所属チームID"])
        return record_state(candidate, kind, income=income, expenses=expenses)

    def salary_player(self, player):
        if (
            self.player(player.name) is None
            and self.opponent_owner(player.name) is None
        ):
            player = next(
                (p for p in self.developed_players if p.name == player.name), player
            )
        if self.salary_mode == SalaryMode.STATIC:
            return player
        record = next((r for r in self.salary_records if r.name == player.name), None)
        return (
            replace(player, monthly_salary=record.monthly_salary) if record else player
        )

    def contract_terms(self, player, kind, short_months=6):
        terms = contract_terms(player, kind, short_months)
        record = next((r for r in self.salary_records if r.name == player.name), None)
        if (
            self.salary_mode == SalaryMode.KD_DYNAMIC
            and record is not None
            and record.fixed
        ):
            return replace(terms, monthly_salary=record.monthly_salary)
        return terms

    @property
    def transfer_multiplier(self):
        return rival_settings.PLAYER_TRANSFER_MULTIPLIER

    def with_salary_mode(self, mode):
        if not self.starter_selection_pending:
            raise SeasonSaveError("給与モードは新規シーズン開始前だけ選択できます。")
        try:
            mode = SalaryMode(mode)
            if mode == self.salary_mode:
                return self
            if mode == SalaryMode.KD_DYNAMIC:
                return replace(
                    self, salary_mode=mode, salary_settings=configured_salary_settings()
                ).with_updated_salaries(initial=True)
            # Switching before start restores the original static snapshots.
            static = {r.name: r.static_salary for r in self.salary_records}

            def restore(player):
                return replace(
                    player,
                    monthly_salary=static.get(player.name, player.monthly_salary),
                )

            clubs = []
            for club in self.opponent_teams:
                players = tuple(restore(p) for p in club.players)
                by_name = {p.name: p for p in players}
                contracts = tuple(
                    (
                        replace(
                            c,
                            monthly_salary=contract_terms(
                                by_name[c.player_name], c.kind, c.duration_months
                            ).monthly_salary,
                        )
                        if self.contract_active(c)
                        else c
                    )
                    for c in club.contracts
                )
                clubs.append(replace(club, players=players, contracts=contracts))
            candidate = replace(
                self,
                salary_mode=mode,
                salary_settings=None,
                salary_records=(),
                salary_updated_month=None,
                starter_candidates=tuple(restore(p) for p in self.starter_candidates),
                opponent_teams=tuple(clubs),
            )
            candidate.validate()
            return candidate
        except SalaryDataError as exc:
            raise SeasonSaveError(str(exc)) from exc

    def with_updated_salaries(self, *, initial=False):
        if (
            self.salary_mode == SalaryMode.STATIC
            or not initial
            and self.salary_updated_month == self.game_month
        ):
            return self
        try:
            pool = {p.name: p for p in all_characters()}
            pool.update(
                (p.name, p) for club in self.opponent_teams for p in club.players
            )
            pool.update(
                (p.name, p) for p in (*self.starter_candidates, *self.owned_players)
            )
            old = {r.name: r for r in self.salary_records}
            static = {name: p.monthly_salary for name, p in pool.items()}
            static.update((name, r.static_salary) for name, r in old.items())
            previous = (
                None
                if initial
                else {
                    name: old[name].monthly_salary if name in old else p.monthly_salary
                    for name, p in pool.items()
                }
            )
            if previous is not None:
                previous.update((name, r.monthly_salary) for name, r in old.items())
            records = salary_records(static, self.salary_settings, previous)
        except (SalaryDataError, OSError) as exc:
            raise SeasonSaveError(str(exc)) from exc
        by_name = {r.name: r for r in records}

        def update_player(p):
            return replace(p, monthly_salary=by_name[p.name].monthly_salary)

        changes = []

        def update_contract(c, team_id=None):
            record = by_name.get(c.player_name)
            if record is None or not self.contract_usable(c, team_id):
                return c
            tenths = {"short": 10, "year1": 10, "year2": 9, "year3": 8}[c.kind]
            salary = (
                record.monthly_salary
                if record.fixed
                else (record.monthly_salary * tenths + 9) // 10
            )
            if not initial and not record.fixed:
                salary = clamp_change(
                    salary, c.monthly_salary, self.salary_settings.max_change
                )
            changes.append((c, salary, record))
            return replace(c, monthly_salary=salary)

        clubs = tuple(
            replace(
                club,
                players=tuple(update_player(p) for p in club.players),
                contracts=tuple(update_contract(c, club.id) for c in club.contracts),
            )
            for club in self.opponent_teams
        )
        candidate = replace(
            self,
            salary_records=records,
            salary_updated_month=self.game_month,
            owned_players=tuple(update_player(p) for p in self.owned_players),
            starter_candidates=tuple(update_player(p) for p in self.starter_candidates),
            contracts=tuple(update_contract(c) for c in self.contracts),
            opponent_teams=clubs,
        )
        candidate.validate()
        for r in records:
            before = old[r.name].monthly_salary if r.name in old else r.static_salary
            log_salary_change(r.name, before, r.monthly_salary, r)
        for c, salary, r in changes:
            log_salary_change(c.player_name, c.monthly_salary, salary, r, contract=True)
        return candidate

    def with_registered_ratings(self):
        records = {r.team_id: r for r in self.ratings}
        for team in (
            SeasonTeam(self.club_id, self.team_name, ()),
            *self.opponent_teams,
        ):
            old = records.get(team.id)
            initial = (
                team.initial_rating
                if isinstance(team, SeasonClub)
                else DEFAULT_TEAM_RATING
            )
            records[team.id] = SeasonRating(
                team.id, team.name, old.value if old else initial
            )
        for run in self.tournaments:
            for team in run.entrants:
                records.setdefault(team.id, SeasonRating(team.id, team.name))
        return replace(self, ratings=tuple(records[key] for key in sorted(records)))

    def rating(self, team_id):
        if self.team(team_id) is not None:
            team_id = self.club_id
        initial = next(
            (t.initial_rating for t in self.opponent_teams if t.id == team_id),
            DEFAULT_TEAM_RATING,
        )
        return next((r.value for r in self.ratings if r.team_id == team_id), initial)

    @property
    def rating_ranking(self):
        active_ids = {self.club_id, *(t.id for t in self.opponent_teams)}
        return tuple(
            sorted(
                (
                    r
                    for r in self.with_registered_ratings().ratings
                    if r.team_id in active_ids
                ),
                key=lambda r: (-r.value, r.team_name.casefold()),
            )
        )

    @property
    def world_rank(self):
        ranking = self.rating_ranking
        rank = next(
            i for i, record in enumerate(ranking, 1) if record.team_id == self.club_id
        )
        return rank, len(ranking)

    @property
    def world_top_percent(self):
        rank, count = self.world_rank
        return rank * 100 / count

    @property
    def world_level_settings(self):
        try:
            current = world_level_for_rank(*self.world_rank)
            return (
                self.world_level_lock
                if self.world_level_lock is not None and self.active_tournaments
                else current
            )
        except WorldLevelError as exc:
            raise SeasonSaveError(f"世界レベル設定が不正です: {exc}") from exc

    @property
    def world_level(self):
        return self.world_level_settings.level

    def enemy_player(self, player, *, familiarity_multiplier=1.0):
        """Effective opponent abilities; persistent ownership and salary use base stats."""
        multiplier = self.world_level_settings.enemy_multiplier
        if familiarity_multiplier != 1.0:
            multiplier *= familiarity_multiplier
        return scale_enemy_player(player, multiplier)

    def match_players(self, players, *, enemy=False):
        """Apply world and pair multipliers together, only to match snapshots."""
        players = tuple(players)
        multiplier = (
            self.pair_metrics(tuple(p.name for p in players))[1]
            if pair_settings.pair_familiarity_enabled
            else 1.0
        )
        return tuple(
            (
                self.enemy_player(p, familiarity_multiplier=multiplier)
                if enemy
                else scale_enemy_player(p, multiplier)
            )
            for p in players
        )

    def displayed_player(self, player):
        return (
            self.enemy_player(player)
            if self.opponent_owner(player.name) is not None
            else player
        )

    def with_rated_result(
        self,
        result_id,
        left_id,
        right_id,
        left_wins,
        right_wins,
        *,
        _continued_contracts=(),
        participants=None,
    ):
        if not isinstance(result_id, str) or not result_id:
            raise SeasonSaveError("レート更新の試合IDが不正です。")
        if result_id in self.rated_results:
            return self
        state = self.with_registered_ratings()
        # Capture the specific preset before its ID is normalized to the club.
        def starters(identifier):
            preset = state.team(identifier)
            if preset is not None:
                return preset.roster
            if identifier == state.club_id:
                preset = state.selected_team or (state.teams[0] if state.teams else None)
                return preset.roster if preset else state.roster or tuple(p.name for p in state.owned_players[:ROSTER_SIZE])
            club = next((c for c in state.opponent_teams if c.id == identifier), None)
            return club.roster if club else ()

        if participants is None:
            participants = {identifier: starters(identifier) for identifier in (left_id, right_id)}
        if not isinstance(participants, dict) or any(
            not isinstance(identifier, str) or not isinstance(names, (tuple, list))
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or len(names) > ROSTER_SIZE
            for identifier, names in participants.items()
        ):
            raise SeasonSaveError("試合の出場選手データが不正です。")
        participants = {state.club_id if state.team(identifier) is not None else identifier: tuple(names)
                        for identifier, names in participants.items()}
        left_id = state.club_id if state.team(left_id) is not None else left_id
        right_id = state.club_id if state.team(right_id) is not None else right_id
        if set(participants) != {left_id, right_id}:
            raise SeasonSaveError("試合の両チームの出場選手を指定してください。")
        ids = {r.team_id for r in state.ratings}
        if left_id == right_id or left_id not in ids or right_id not in ids:
            raise SeasonSaveError("レート更新の対戦チームが不正です。")
        if (
            type(left_wins) is not int
            or type(right_wins) is not int
            or min(left_wins, right_wins) < 0
            or left_wins == right_wins
        ):
            raise SeasonSaveError("レート更新には勝敗が確定したマップ数が必要です。")
        left, right = series_ratings(
            state.rating(left_id), state.rating(right_id), left_wins, right_wins
        )
        state = replace(
            state,
            ratings=tuple(
                (
                    replace(r, value=left if r.team_id == left_id else right)
                    if r.team_id in (left_id, right_id)
                    else r
                )
                for r in state.ratings
            ),
            rated_results=(*state.rated_results, result_id),
        )
        state = state._with_match_loyalty(
            left_id, right_id, left_wins > right_wins, _continued_contracts, participants
        )
        from season_contract_endings import settle_contract_endings
        state = settle_contract_endings(state, _continued_contracts)
        state.validate()
        return state._record_history("レート更新")

    def _with_match_loyalty(self, left_id, right_id, left_won, continued_contracts=(), participants=None):
        outcomes = {left_id: left_won, right_id: not left_won}

        def updated_contracts(team_id, players, contracts):
            if team_id not in outcomes:
                return contracts
            members = {p.name: p for p in players}
            updated = []
            for contract in contracts:
                player = members.get(contract.player_name)
                if player is None or not (
                    self.contract_usable(contract, team_id)
                    or (team_id, contract.player_name) in continued_contracts
                ):
                    updated.append(contract)
                    continue
                delta = (1.0 if outcomes[team_id] else -(10 - player.loyalty) / 10
                         ) if player.name in participants[team_id] else -BENCHED_LOYALTY_LOSS
                updated.append(
                    replace(
                        contract, team_loyalty=round(contract.team_loyalty + delta, 10)
                    )
                    if delta
                    else contract
                )
            return tuple(updated)

        return replace(
            self,
            contracts=updated_contracts(
                self.club_id, self.owned_players, self.contracts
            ),
            opponent_teams=tuple(
                (
                    replace(
                        club,
                        contracts=updated_contracts(
                            club.id, club.players, club.contracts
                        ),
                    )
                    if club.id in outcomes
                    else club
                )
                for club in self.opponent_teams
            ),
        ).with_resolved_transfer_offers()

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
        return self.world_level_settings.sponsor_funds

    @property
    def monthly_payroll(self):
        return sum(c.monthly_salary for c in self.contracts if self.contract_usable(c))

    def with_sponsor_contract(self, active):
        if type(active) is not bool:
            raise SeasonSaveError("スポンサー契約の状態が不正です。")
        candidate = replace(self, sponsor_active=active)
        candidate.validate()
        return candidate._record_history("スポンサー契約変更")

    @property
    def scout_players(self):
        # Saved ability and salary snapshots take priority over the live catalog.
        players = {p.name: p for p in all_characters()}
        players.update((p.name, p) for p in self.developed_players)
        players.update(
            (p.name, p) for club in self.opponent_teams for p in club.players
        )
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
        return int(
            (
                Decimal(player.monthly_salary) * Decimal(str(owner.transfer_multiplier))
            ).to_integral_value(rounding=ROUND_CEILING)
        )

    def recruitment_blocked(self, name):
        if self.player(name) is not None:
            return "すでに所持している選手です。再契約は契約状況から行ってください。"
        if any(
            not run.completed
            and any(p.name == name for t in run.entrants for p in t.players)
            for run in self.tournaments
        ):
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
        if (
            len(names) != ROSTER_SIZE
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != ROSTER_SIZE
        ):
            raise SeasonSaveError("初期キャラは異なる5人を選んでください。")
        pool = {player.name: player for player in self.starter_candidates}
        if not set(names).issubset(pool):
            raise SeasonSaveError("初期キャラの候補から選んでください。")
        players = tuple(pool[name] for name in names)
        candidate = replace(
            self,
            starter_selection_pending=False,
            starter_selection=tuple(names),
            owned_players=players,
            contracts=tuple(
                signed_contract(p, self.contract_terms(p, "short", 6), self.game_month,
                                state=self, team_id=self.club_id)
                for p in players
            ),
            opponent_teams=resolve_club_memberships(self.opponent_teams, names),
        )
        candidate = with_randomized_clubs(candidate)
        candidate = initial_pair_days(candidate)
        candidate.validate()
        return candidate.with_updated_salaries(initial=True)._record_history(
            "初期選手確定"
        )

    @property
    def date(self):
        return (
            parse_date(self.game_date)
            if self.game_date is not None
            else add_months(parse_date(self.start_date), self.game_month)
        )

    @property
    def phase(self):
        return phase_for(self.date, self.in_season_periods)

    def tournament(self, event_id):
        return next(
            (run for run in self.tournaments if run.tournament_id == event_id), None
        )

    def tournament_definition(self, event_id):
        return next(
            (event for event in self.tournament_definitions if event.id == event_id),
            None,
        )

    @property
    def active_tournaments(self):
        return tuple(
            run
            for run in self.tournaments
            if not run.completed
            and self.tournament_definition(run.tournament_id) is not None
            and parse_date(self.tournament_definition(run.tournament_id).start_date)
            <= self.date
        )

    def _with_tournament_world_level(self):
        if not self.active_tournaments:
            if self.world_level_lock is not None or any(
                c.world_level_lock is not None for c in self.opponent_teams
            ):
                return replace(
                    self,
                    world_level_lock=None,
                    opponent_teams=tuple(
                        replace(c, world_level_lock=None) for c in self.opponent_teams
                    ),
                )
            return self
        ranking = self.rating_ranking
        ranks = {r.team_id: i for i, r in enumerate(ranking, 1)}
        clubs = tuple(
            (
                replace(
                    c, world_level_lock=world_level_for_rank(ranks[c.id], len(ranking))
                )
                if c.world_level_lock is None
                else c
            )
            for c in self.opponent_teams
        )
        if self.world_level_lock is None:
            return replace(
                self, world_level_lock=self.world_level_settings, opponent_teams=clubs
            )
        return (
            replace(self, opponent_teams=clubs)
            if clubs != self.opponent_teams
            else self
        )

    @property
    def visible_tournaments(self):
        return tuple(
            event
            for event in self.tournament_definitions
            if self.tournament(event.id) is not None or event.appears(self)
        )

    @property
    def entry_deadline_tournaments(self):
        """Unentered events that can still be joined before leaving today."""
        return tuple(
            event
            for event in self.visible_tournaments
            if event.allow_player_entry
            and self.date == parse_date(event.start_date)
            and (
                self.tournament(event.id) is None or self.tournament(event.id).declined
            )
        )

    @property
    def pending_tournaments(self):
        return tuple(
            event
            for event in self.visible_tournaments
            if parse_date(event.start_date) <= self.date
            and (
                self.tournament(event.id) is None
                and event.allow_player_entry
                and self.date <= parse_date(event.end_date)
                or self.tournament(event.id) is not None
                and not self.tournament(event.id).completed
            )
        )

    def check_tournament_match_day(self, event_id):
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        if event is None or run is None or run.completed:
            raise SeasonSaveError("進行中の大会を選択してください。")
        if self.date < parse_date(event.start_date):
            raise SeasonSaveError("大会試合は開始日以降に行ってください。")
        if run.last_match_date is not None and self.date <= parse_date(
            run.last_match_date
        ):
            raise SeasonSaveError(
                "大会は1日1試合です。1日進めてから次のシリーズを開始してください。"
            )

    def with_tournament_entry(
        self, event_id, team_id=None, *, own_ai=None, igl=None, carrier=None
    ):
        event = self.tournament_definition(event_id)
        if (
            event is None
            or not event.appears(self)
            or self.date > parse_date(event.start_date)
        ):
            raise SeasonSaveError("大会は未出現または参加登録の締切を過ぎています。")
        if (
            self.tournament(event_id) is not None
            and not self.tournament(event_id).declined
        ):
            raise SeasonSaveError("この大会は参加判断済みです。")
        own = self.team(team_id) if event.allow_player_entry else None
        if event.allow_player_entry and (
            own is None or any(not self.can_play(name) for name in own.roster)
        ):
            raise SeasonSaveError("契約中の5人を編成した参加チームを選択してください。")
        if own and any(
            not r.completed and r.own_team_id is not None for r in self.tournaments
        ):
            raise SeasonSaveError(
                "参加中の大会を完了してから次の大会へ参加してください。"
            )
        entrants = []
        if own:
            players = tuple(self.player(name) for name in own.roster)
            entrants.append(
                CompetitionTeam(
                    self.club_id,
                    self.team_name,
                    players,
                    own.ai if own_ai is None else own_ai,
                    igl or own.igl or max(players, key=lambda p: p.iq).name,
                    carrier or own.carrier or players[0].name,
                )
            )
        rivals = self.tournament_rivals(event)
        needed = event.team_count - len(entrants)
        if len(rivals) + len(entrants) < 2:
            # There is no opponent to substitute yet. Leave registration open
            # until the deadline instead of creating an invalid bracket.
            return self
        entrants.extend(
            CompetitionTeam(
                club.id,
                club.name,
                club.players[:ROSTER_SIZE],
                club.ai,
                club.effective_igl,
                club.effective_carrier,
            )
            for club in rivals[:needed]
        )
        run = TournamentProgress(
            event.id,
            self.club_id if own else None,
            tuple(entrants),
            seed=int(uuid4().hex[:8], 16) % (2**31),
            preset_id=own.id if own else None,
        )
        candidate = replace(
            self,
            tournaments=tuple(
                r for r in self.tournaments if r.tournament_id != event_id
            )
            + (run,),
        )._with_tournament_world_level()
        candidate.validate()
        return candidate._record_history("大会参加登録")

    def tournament_rivals(self, event):
        """Prefer configured invitees, replacing unavailable clubs from the league."""
        eligible = tuple(
            c for c in self.opponent_teams if len(c.players) >= ROSTER_SIZE
        )
        by_name = {c.name: c for c in eligible}
        preferred = tuple(by_name[n] for n in event.opponent_teams if n in by_name)
        chosen = {c.id for c in preferred}
        return preferred + tuple(c for c in eligible if c.id not in chosen)

    def with_declined_tournament(self, event_id):
        event = self.tournament_definition(event_id)
        if (
            event is None
            or not event.appears(self)
            or self.date > parse_date(event.start_date)
        ):
            raise SeasonSaveError("大会は未出現または参加登録の締切を過ぎています。")
        if not event.participation_optional or not event.allow_player_entry:
            raise SeasonSaveError("この大会では不参加を選択できません。")
        if self.tournament(event_id) is not None:
            raise SeasonSaveError("この大会は参加判断済みです。")
        candidate = replace(
            self,
            tournaments=(
                *self.tournaments,
                TournamentProgress(event_id, None, completed=True, declined=True),
            ),
        )
        candidate.validate()
        return candidate._record_history("大会不参加")

    def with_tournament_result(self, event_id, score):
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        if event is None or run is None or run.completed:
            raise SeasonSaveError("結果を登録する進行中の大会が見つかりません。")
        self.check_tournament_match_day(event_id)
        candidate = self._with_tournament_score(event, run, score)
        if run.own_team_id not in (score.left_id, score.right_id):
            candidate = (
                replace(candidate, day_advance_pending=True)
                if candidate.entry_deadline_tournaments
                else candidate.advance_days()
            )
        updated = candidate.tournament(event_id)
        if not updated.completed and player_eliminated(event, updated):
            return candidate.with_tournament_rating_finish(event_id)
        return candidate

    def _with_tournament_score(self, event, run, score):
        """Record one series, including its rating update and completion prize."""
        event_id = event.id
        continued_contracts = self.deferred_contracts
        updated = replace(
            run, results=(*run.results, score), last_match_date=self.date.isoformat()
        )
        pending, ranking = next_match(event, updated)
        prize = 0
        prizes = {}
        if pending is None:
            prizes = {
                team_id: event.prizes.get(rank, 0)
                for rank, team_id in enumerate(ranking, 1)
            }
            if run.own_team_id is not None:
                rank = ranking.index(run.own_team_id) + 1
                prize = event.prizes.get(rank, 0)
            updated = replace(
                updated,
                completed=True,
                ranking=ranking,
                prize_paid=prize,
                completed_date=self.date.isoformat(),
            )
        candidate = replace(
            self,
            money=self.money + prize,
            opponent_teams=tuple(
                replace(c, money=c.money + prizes.get(c.id, 0))
                for c in self.opponent_teams
            ),
            tournaments=tuple(
                updated if r.tournament_id == event_id else r for r in self.tournaments
            ),
        )
        candidate.validate()
        candidate = candidate._record_history(
            "大会試合結果", income=(cash_item("大会賞金", prize),) if prize else ()
        )
        candidate = candidate.with_rated_result(
            f"tournament:{event_id}:{score.match_id}",
            score.left_id,
            score.right_id,
            score.left_wins,
            score.right_wins,
            _continued_contracts=continued_contracts,
            participants={t.id: tuple(p.name for p in t.players) for t in run.entrants
                          if t.id in (score.left_id, score.right_id)},
        )
        candidate = candidate._with_tournament_world_level()
        if updated.completed:
            from season_monthly_events import settle_deferred_contracts

            candidate = settle_deferred_contracts(
                candidate, continued_contracts, event_id
            )
        return candidate

    def _tournament_rating_score(self, run, match):
        if run.own_team_id in (match.left, match.right):
            raise SeasonSaveError("自チームの試合をレート判定することはできません。")
        # Stable across save/reload and failed save retries; draw each map
        # independently until one side reaches the required number of wins.
        rng = Random((run.seed + len(run.results) * 1000) % (2**31))
        left_wins = right_wins = 0
        while max(left_wins, right_wins) < match.maps_to_win:
            probability = expected_score(self.rating(match.left), self.rating(match.right))
            if rng.random() < probability:
                left_wins += 1
            else:
                right_wins += 1
        return SeriesScore(
            match.id,
            match.left,
            match.right,
            left_wins,
            right_wins,
            decided_by_rating=True,
        )

    def with_tournament_rating_result(self, event_id):
        """Resolve today's NPC series using the current Elo win probability."""
        self.check_tournament_match_day(event_id)
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        match, _ = next_match(event, run)
        return self.with_tournament_result(
            event_id, self._tournament_rating_score(run, match)
        )

    def with_tournament_rating_finish(self, event_id):
        """Resolve remaining NPC cards while consuming one calendar day each."""
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        if event is None or run is None:
            raise SeasonSaveError("進行中の大会を選択してください。")
        if run.completed:
            return self
        if not player_eliminated(event, run):
            raise SeasonSaveError(
                "自チームの敗退が確定してからレート判定で終了できます。"
            )
        candidate = self
        while not run.completed:
            if run.last_match_date == candidate.game_date:
                following = candidate.advance_days(stop_at_entry_deadlines=True)
                if following.date == candidate.date:
                    break
                candidate = following
                run = candidate.tournament(event_id)
            candidate.check_tournament_match_day(event_id)
            match, _ = next_match(event, run)
            score = candidate._tournament_rating_score(run, match)
            candidate = candidate._with_tournament_score(event, run, score)
            following = candidate.advance_days(stop_at_entry_deadlines=True)
            if following.date == candidate.date:
                break
            candidate = following
            run = candidate.tournament(event_id)
        return candidate

    def with_tournament_forfeit(self, event_id):
        event, run = self.tournament_definition(event_id), self.tournament(event_id)
        if event is None or run is None or run.completed:
            raise SeasonSaveError("進行中の大会を選択してください。")
        match, _ = next_match(event, run)
        if run.own_team_id not in (match.left, match.right):
            raise SeasonSaveError("自分のチームのシリーズだけ棄権できます。")
        score = SeriesScore(
            match.id,
            match.left,
            match.right,
            0 if run.own_team_id == match.left else match.maps_to_win,
            0 if run.own_team_id == match.right else match.maps_to_win,
        )
        return self.with_tournament_result(event_id, score)

    def contract(self, name):
        return next((c for c in self.contracts if c.player_name == name), None)

    def contract_active(self, contract):
        return contract.active(self.date, self.start_date)

    def tournament_reserves_contract(self, contract, team_id=None):
        team_id = self.club_id if team_id is None else team_id
        if contract.end_reason is not None or contract.starts_on(self.start_date) > self.date:
            return False
        if team_id == self.club_id:
            if self.player(contract.player_name) is None:
                return False
        elif not any(
            c.id == team_id and contract.player_name in c.members
            for c in self.opponent_teams
        ):
            return False
        expiry = contract.ends_on(self.start_date)
        return any(
            (
                self.contract_active(contract)
                or expiry
                > parse_date(self.tournament_definition(run.tournament_id).start_date)
            )
            and any(
                t.id == team_id
                and any(p.name == contract.player_name for p in t.players)
                for t in run.entrants
            )
            for run in self.active_tournaments
        )

    def contract_end_deferred(self, contract, team_id=None):
        ending = self.date >= contract.ends_on(self.start_date) or (
            contract.kind == "short"
            and contract.team_loyalty < 0
            and self.game_month > 0
        )
        return ending and self.tournament_reserves_contract(contract, team_id)

    def contract_usable(self, contract, team_id=None):
        return self.contract_active(contract) or self.contract_end_deferred(
            contract, team_id
        )

    @property
    def deferred_contracts(self):
        return frozenset(
            (team_id, c.player_name)
            for team_id, contracts in (
                (self.club_id, self.contracts),
                *((t.id, t.contracts) for t in self.opponent_teams),
            )
            for c in contracts
            if self.contract_end_deferred(c, team_id)
        )

    def can_play(self, name):
        contract = self.contract(name)
        return (
            self.player(name) is not None
            and contract is not None
            and self.contract_usable(contract)
        )

    @property
    def unplayable_roster(self):
        return tuple(name for name in self.roster if not self.can_play(name))

    def with_playable_roster(self):
        """Repair the editable lineup, retaining saved presets for replacement."""
        if not self.unplayable_roster:
            return self
        return self.with_roster(
            tuple(name for name in self.roster if self.can_play(name))
        )

    @property
    def lft_players(self):
        affiliated = {p.name for p in self.owned_players}
        affiliated.update(p.name for team in self.opponent_teams for p in team.players)
        return tuple(p for p in self.scout_players if p.name not in affiliated)

    def training_terms(self, name, kind):
        player = self.player(name)
        if self.starter_selection_pending or player is None:
            raise SeasonSaveError("育成する所持選手を選択してください。")
        return training_terms(player, kind)

    def with_trained_player(self, name, kind, *, advance_day=True):
        self.check_day_action()
        terms = self.training_terms(name, kind)
        if self.money < terms.cost:
            raise SeasonSaveError(
                f"{terms.title}には{terms.cost:,}円が必要です（現在{self.money:,}円）。"
            )
        player = replace(
            self.player(name),
            **{terms.level_field: terms.next_level, terms.stat_field: terms.after},
        )
        candidate = replace(
            self,
            money=self.money - terms.cost,
            owned_players=tuple(
                player if p.name == name else p for p in self.owned_players
            ),
            developed_players=tuple(p for p in self.developed_players if p.name != name)
            + (player,),
        )
        candidate.validate()
        candidate = candidate._record_history(
            terms.title, expenses=(cash_item(terms.title, terms.cost, name),)
        )
        return candidate._finish_day_action(advance_day)

    def with_scrim_result(
        self,
        result_id,
        own_id,
        opponent_id,
        own_wins,
        opponent_wins,
        *,
        advance_day=True,
        participants=None,
    ):
        if result_id in self.rated_results:
            return self
        self.check_day_action()
        candidate = self.with_rated_result(
            result_id, own_id, opponent_id, own_wins, opponent_wins, participants=participants
        )
        # Scrims share the calendar and tournament restrictions with training.
        return candidate._finish_day_action(advance_day)

    def with_scouted_player(self, name, kind, short_months=6, *, advance_day=True):
        blocked = self.scout_blocked() or self.recruitment_blocked(name)
        if blocked:
            raise SeasonSaveError(blocked)
        player = next((p for p in self.scout_players if p.name == name), None)
        if player is None:
            raise SeasonSaveError("スカウトする選手が見つかりません。")
        if self.contract_refused(name):
            raise SeasonSaveError(
                "このチームとは再契約できない選手です（忠誠が負、または永久拒否）。"
            )
        fee = self.transfer_fee(name)
        terms = self.contract_terms(player, kind, short_months)
        if self.money < fee + terms.total_required_funds:
            raise SeasonSaveError(
                f"契約には移籍金{fee:,}円・契約金{terms.signing_bonus:,}円と契約条件の資金{terms.required_funds:,}円が必要です（現在{self.money:,}円）。"
            )
        owner = self.opponent_owner(name)
        clubs = []
        for club in self.opponent_teams:
            if club != owner:
                clubs.append(club)
                continue
            remaining = tuple(p for p in club.players if p.name != name)
            roster = {p.name for p in remaining[:ROSTER_SIZE]}
            clubs.append(
                replace(
                    club,
                    players=remaining,
                    money=club.money + fee,
                    acquired_members=tuple(
                        n for n in club.acquired_members if n != name
                    ),
                    igl=club.igl if club.igl in roster else None,
                    carrier=club.carrier if club.carrier in roster else None,
                    contracts=tuple(
                        (
                            replace(c, end_reason="released")
                            if c.player_name == name
                            else c
                        )
                        for c in club.contracts
                    ),
                )
            )
        candidate = replace(
            self,
            money=self.money - fee,
            opponent_teams=tuple(clubs),
            transferred_players=(
                (*self.transferred_players, name)
                if owner and name not in self.transferred_players
                else self.transferred_players
            ),
        )
        candidate = candidate._with_scout_use(self.club_id)._with_signed_contract(
            player, kind, short_months, recruit=True, transfer_fee=fee
        )
        return candidate._finish_day_action(advance_day)

    def _finish_day_action(self, advance_day):
        if advance_day:
            return self._with_next_day()
        candidate = replace(self, day_advance_pending=True)
        candidate.validate()
        return candidate

    def with_renewed_contract(self, name, kind, short_months=6):
        if self.contract_refused(name):
            raise SeasonSaveError("忠誠が負になり、このチームとの再契約を永久に断られました。")
        player = self.player(name)
        old = self.contract(name)
        if player is None or old is None:
            raise SeasonSaveError("再契約する所持選手が見つかりません。")
        if self.contract_end_deferred(old):
            raise SeasonSaveError(
                "出場中につき契約延期中です。大会終了後に再契約できます。"
            )
        if self.contract_active(old):
            raise SeasonSaveError("再契約は現在の契約が終了してから行えます。")
        if self.contract_refused(name):
            raise SeasonSaveError("このチームとの再契約を永久に断られました。")
        return self._with_signed_contract(player, kind, short_months, recruit=False)

    def _with_signed_contract(
        self, player, kind, short_months, recruit, *, transfer_fee=0
    ):
        terms = self.contract_terms(player, kind, short_months)
        if self.money < terms.total_required_funds:
            raise SeasonSaveError(
                f"契約には契約金{terms.signing_bonus:,}円と契約条件の資金{terms.required_funds:,}円が必要です（現在{self.money:,}円）。"
            )
        loyalty, signing_number = signing_loyalty(self, player.name, self.club_id, kind)
        contract = PlayerContract(
            player.name,
            kind,
            terms.monthly_salary,
            self.game_month,
            terms.months,
            loyalty,
            signing_number=signing_number,
            signed_on=self.date.isoformat(),
        )
        candidate = replace(
            self,
            money=self.money - terms.signing_bonus,
            owned_players=(
                (*self.owned_players, player) if recruit else self.owned_players
            ),
            contracts=tuple(c for c in self.contracts if c.player_name != player.name)
            + (contract,),
        )
        candidate.validate()
        costs = (cash_item("契約金", terms.signing_bonus, player.name),)
        if transfer_fee:
            costs += (cash_item("移籍金", transfer_fee, player.name),)
        return candidate._record_history(
            "選手獲得" if recruit else "契約更新", expenses=costs
        )

    def with_team_loyalty(self, name, value, team_id=None):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise SeasonSaveError("チームへの忠誠には有限の数値を設定してください。")
        team_id = self.club_id if team_id is None else team_id
        state = remember_loyalties(self)
        if (
            name not in state.team_loyalties
            or team_id not in state.team_loyalties[name]
        ):
            raise SeasonSaveError("忠誠を変更する選手またはチームが見つかりません。")
        player = next((p for p in self.scout_players if p.name == name), None)
        if (
            player is not None
            and player.loyalty == 10
            and value < self.team_loyalty(name, team_id)
        ):
            return self
        memory = {n: dict(teams) for n, teams in state.team_loyalties.items()}
        memory[name][team_id] = value
        candidate = replace(
            state,
            team_loyalties=memory,
            contracts=tuple(
                (
                    replace(c, team_loyalty=value)
                    if team_id == self.club_id and c.player_name == name
                    else c
                )
                for c in self.contracts
            ),
            opponent_teams=tuple(
                (
                    replace(
                        club,
                        contracts=tuple(
                            (
                                replace(c, team_loyalty=value)
                                if c.player_name == name
                                else c
                            )
                            for c in club.contracts
                        ),
                    )
                    if club.id == team_id
                    else club
                )
                for club in self.opponent_teams
            ),
        )
        candidate = candidate.with_resolved_transfer_offers()
        from season_contract_endings import settle_contract_endings
        candidate = settle_contract_endings(candidate)
        candidate.validate()
        return candidate

    @property
    def pending_transfer_offers(self):
        return tuple(o for o in self.transfer_offers if o.status == "pending")

    def transfer_offer(self, name):
        return next(
            (o for o in self.pending_transfer_offers if o.player_name == name), None
        )

    def with_transfer_response(self, offer_id, accept):
        if type(accept) is not bool:
            raise SeasonSaveError("オファーへの回答が不正です。")
        offer = next(
            (o for o in self.pending_transfer_offers if o.id == offer_id), None
        )
        if offer is None:
            raise SeasonSaveError("回答する移籍オファーが見つかりません。")
        # Threshold takes priority even if the caller attempts to decline.
        if self.contract(offer.player_name).team_loyalty < FORCED_OFFER_LOYALTY:
            candidate = self._with_offer_transfer(offer, "forced")
        elif accept:
            candidate = self._with_offer_transfer(offer, "accepted")
        else:
            candidate = replace(
                self,
                transfer_offers=tuple(
                    replace(o, status="rejected") if o.id == offer_id else o
                    for o in self.transfer_offers
                ),
            )
        candidate.validate()
        return candidate._record_history("移籍オファー回答")

    def with_resolved_transfer_offers(self):
        candidate = self
        latest = {offer.player_name: offer for offer in self.transfer_offers}
        clubs = {club.id for club in self.opponent_teams}
        for offer in latest.values():
            if (
                offer.status not in ("pending", "rejected")
                or offer.team_id not in clubs
            ):
                continue
            contract = candidate.contract(offer.player_name)
            if candidate.player(offer.player_name) is None:
                candidate = replace(
                    candidate,
                    transfer_offers=tuple(
                        replace(o, status="cancelled") if o.id == offer.id else o
                        for o in candidate.transfer_offers
                    ),
                )
            elif contract.team_loyalty < FORCED_OFFER_LOYALTY:
                candidate = candidate._with_offer_transfer(offer, "forced")
        return candidate

    def _with_offer_transfer(self, offer, status):
        self = remember_loyalties(self)
        player = self.player(offer.player_name)
        buyer = next(c for c in self.opponent_teams if c.id == offer.team_id)
        terms = offer_terms(self, offer)
        if (
            self.contract_refused(player.name, buyer.id)
            or self.scout_blocked(buyer.id)
            or not can_sign(self, buyer, terms, offer.fee, exclude=offer.id)
        ):
            # Do not create a transfer or a fee without a solvent buyer.
            return replace(
                self,
                transfer_offers=tuple(
                    replace(o, status="cancelled") if o.id == offer.id else o
                    for o in self.transfer_offers
                ),
            )
        candidate = self._with_scout_use(buyer.id)._with_departed_player(
            player.name, record=False
        )
        clubs = []
        for club in candidate.opponent_teams:
            if club.id == offer.team_id:
                players = [*club.players, player]
                reserved = {
                    p.name
                    for run in self.tournaments
                    if not run.completed
                    for t in run.entrants
                    for p in t.players
                }
                if not any(p.name in reserved for p in club.players[:ROSTER_SIZE]):
                    from season_monthly_events import player_strength

                    order = {name: i for i, name in enumerate(club.regular_members)}
                    acquired = set(club.acquired_members) | (
                        {player.name} if player.name not in order else set()
                    )
                    if acquired:
                        players.sort(key=lambda p: -player_strength(p))
                    else:
                        players.sort(key=lambda p: order.get(p.name, len(order)))
                    players = [
                        p
                        for i, p in enumerate(players)
                        if i < max(ROSTER_SIZE, len(order))
                        or p.name in reserved
                        or p.name in acquired
                        or acquired
                        and p.name in order
                    ]
                members = {p.name for p in players}
                roster = {p.name for p in players[:ROSTER_SIZE]}
                club = replace(
                    club,
                    players=tuple(players),
                    money=club.money - offer.fee - terms.signing_bonus,
                    acquired_members=(
                        tuple(dict.fromkeys((*club.acquired_members, player.name)))
                        if player.name not in club.regular_members
                        else club.acquired_members
                    ),
                    contracts=tuple(
                        (
                            replace(c, end_reason="released")
                            if c.player_name not in members and c.end_reason is None
                            else c
                        )
                        for c in club.contracts
                        if c.player_name != player.name
                    )
                    + (
                        signed_contract(
                            player,
                            terms,
                            self.game_month,
                            state=self, team_id=club.id,
                        ),
                    ),
                    igl=(
                        club.regular_igl
                        if club.regular_igl in roster
                        else club.igl if club.igl in roster else None
                    ),
                    carrier=(
                        club.regular_carrier
                        if club.regular_carrier in roster
                        else club.carrier if club.carrier in roster else None
                    ),
                )
            clubs.append(club)
        return replace(
            candidate,
            opponent_teams=tuple(clubs),
            money=candidate.money + offer.fee,
            transfer_offers=tuple(
                replace(o, status=status) if o.id == offer.id else o
                for o in candidate.transfer_offers
            ),
        )._record_history(
            "強制移籍" if status == "forced" else "選手移籍",
            income=(cash_item("移籍金", offer.fee, player.name),),
        )

    def advance_months(
        self, months=1, *, pay_salaries=True, stop_for_tournaments=False
    ):
        if type(months) is not int or months < 1:
            raise SeasonSaveError("進める月数は1以上の整数で指定してください。")
        target = add_months(self.date, months)
        return self.advance_days(
            (target - self.date).days,
            pay_salaries=pay_salaries,
            stop_for_tournaments=stop_for_tournaments,
        )

    def advance_days(
        self,
        days=1,
        *,
        pay_salaries=True,
        stop_for_tournaments=True,
        stop_at_entry_deadlines=False,
    ):
        if type(days) is not int or days < 1:
            raise SeasonSaveError("進める日数は1以上の整数で指定してください。")
        candidate = self._with_calendar_tournaments()
        for _ in range(days):
            if stop_at_entry_deadlines and candidate.entry_deadline_tournaments:
                break
            pending = candidate.pending_tournaments
            # After today's series, allow exactly one day forward, then stop at
            # the next unplayed series. Month-end processing follows the same path.
            blocking = [
                e
                for e in pending
                if candidate.tournament(e.id) is not None
                and candidate.tournament(e.id).own_team_id == candidate.club_id
                and candidate.tournament(e.id).last_match_date
                != candidate.date.isoformat()
            ]
            if blocking:
                break
            candidate = candidate._with_next_day(pay_salaries=pay_salaries)
        candidate.validate()
        return candidate

    def _with_calendar_tournaments(self, *, close_registration=False):
        """Keep entry open on the start date until leaving it; play NPC series daily."""
        if self.starter_selection_pending:
            return self
        definitions = extend_annual_calendar(self.tournament_definitions, self.date, parse_date(self.start_date))
        candidate = self if definitions == self.tournament_definitions else replace(self, tournament_definitions=definitions)
        for event in definitions:
            if candidate.date < parse_date(event.start_date):
                continue
            run = candidate.tournament(event.id)
            if run is None or run.declined:
                if (
                    not close_registration
                    and candidate.date == parse_date(event.start_date)
                    and event.allow_player_entry
                    and event.appears(candidate)
                ):
                    continue
                rivals = candidate.tournament_rivals(event)
                # Use the available clubs when the configured field is larger
                # than the league; no fictional duplicate clubs/players.
                if len(rivals) < 2:
                    continue
                entrants = tuple(
                    CompetitionTeam(
                        c.id,
                        c.name,
                        c.players[:ROSTER_SIZE],
                        c.ai,
                        c.effective_igl,
                        c.effective_carrier,
                    )
                    for c in rivals[: event.team_count]
                )
                run = TournamentProgress(
                    event.id,
                    None,
                    entrants,
                    seed=uuid5(
                        NAMESPACE_URL, f"season-npc:{candidate.start_date}:{event.id}"
                    ).int
                    % (2**31),
                )
                candidate = replace(
                    candidate,
                    tournaments=tuple(
                        r for r in candidate.tournaments if r.tournament_id != event.id
                    )
                    + (run,),
                )._with_tournament_world_level()
                candidate.validate()
                candidate = candidate._record_history("大会自動開催（自チーム不参加）")
            if (
                run.own_team_id is None
                and not run.completed
                and run.last_match_date != candidate.date.isoformat()
            ):
                match, _ = next_match(event, run)
                candidate = candidate._with_tournament_score(
                    event, run, candidate._tournament_rating_score(run, match)
                )
        return candidate

    def _with_next_day(self, *, pay_salaries=True):
        # Old saves may already be on an unregistered event's start date.
        # Resolve that date too, using the same path as explicit day advances.
        current = self._with_calendar_tournaments(close_registration=True)
        if current is not self:
            return current._with_next_day(pay_salaries=pay_salaries)
        # A day belongs to the contracts active at its beginning, including the
        # last day before expiry or a month-end departure/transfer.
        active = {c.player_name for c in self.contracts if self.contract_usable(c)}
        grown = tuple(
            (
                replace(p, iq=float(Decimal(str(p.iq)) + Decimal("0.1")))
                if p.name in active
                else p
            )
            for p in self.owned_players
        )
        developed = {p.name: p for p in self.developed_players}
        developed.update((p.name, p) for p in grown if p.name in active)
        candidate = replace(
            self,
            owned_players=grown,
            developed_players=tuple(developed.values()),
            pair_news_deferred=True,
            day_advance_pending=False,
        )
        day = candidate.date + timedelta(days=1)
        index = month_index(parse_date(candidate.start_date), day)
        new_month = index != candidate.game_month
        if new_month:
            income = candidate.monthly_sponsor_income
            candidate = monthly_settlement(candidate, pay_salaries=pay_salaries)
            # Normal calendar advances and completed scrims share monthly settlement.
            players = {p.name: p for p in candidate.owned_players}
            candidate = replace(
                candidate,
                contracts=tuple(
                    (
                        replace(
                            c,
                            team_loyalty=round(
                                c.team_loyalty
                                - (10 - players[c.player_name].loyalty) / 10,
                                10,
                            ),
                        )
                        if candidate.contract_usable(c)
                        and c.player_name in players
                        and players[c.player_name].loyalty < 10
                        else c
                    )
                    for c in candidate.contracts
                ),
            )
            candidate = candidate.with_resolved_transfer_offers()
            for contract in candidate.contracts:
                if (
                    contract.kind == "short"
                    and candidate.contract_active(contract)
                    and contract.team_loyalty < 0
                    and not candidate.tournament_reserves_contract(contract)
                ):
                    candidate = candidate._with_departed_player(contract.player_name)
            payroll = candidate.monthly_payroll if pay_salaries else 0
            wages = tuple(
                cash_item("給与", c.monthly_salary, c.player_name)
                for c in candidate.contracts
                if pay_salaries and candidate.contract_usable(c)
            )
            candidate = replace(
                candidate, money=candidate.money - payroll + income, game_month=index
            )
        candidate = replace(candidate, game_date=day.isoformat())
        from season_contract_endings import settle_contract_endings
        candidate = settle_contract_endings(candidate)
        if new_month:
            candidate = candidate._record_history(
                "月次決算",
                income=(cash_item("スポンサー収入", income),) if income else (),
                expenses=wages,
            )
            candidate = candidate.with_updated_salaries()
            candidate = process_monthly_events(candidate)
            candidate = settle_contract_endings(candidate)
        candidate = advance_pair_days(candidate)
        candidate = replace(daily_news(candidate, self), pair_news_deferred=False)
        if new_month:
            from season_pair_familiarity import history_metrics

            candidate = replace(
                candidate,
                history=tuple(
                    (
                        {**entry, "ペア練度": history_metrics(candidate)}
                        if entry["種別"] == "月次決算"
                        and entry["ゲーム内日付"] == candidate.date.isoformat()
                        else entry
                    )
                    for entry in candidate.history
                ),
            )
        candidate = candidate._with_calendar_tournaments()
        candidate = candidate._with_tournament_world_level()
        candidate.validate()
        return candidate._record_history("日付進行")

    def _with_departed_player(self, name, *, record=True):
        # Registered lineups must always have five players. Keep the remaining draft
        # so the user can replace a departing player and register the team again.
        teams = tuple(t for t in self.teams if name not in t.roster)
        team_ids = {t.id for t in teams}
        candidate = replace(
            self,
            owned_players=tuple(p for p in self.owned_players if p.name != name),
            roster=tuple(n for n in self.roster if n != name),
            teams=teams,
            preset_igl=None if self.preset_igl == name else self.preset_igl,
            preset_carrier=None if self.preset_carrier == name else self.preset_carrier,
            editing_team_id=(
                self.editing_team_id if self.editing_team_id in team_ids else None
            ),
            selected_team_id=(
                self.selected_team_id if self.selected_team_id in team_ids else None
            ),
            contracts=tuple(
                replace(c, end_reason="left") if c.player_name == name else c
                for c in self.contracts
            ),
            transfer_offers=tuple(
                (
                    replace(o, status="cancelled")
                    if o.player_name == name and o.status in ("pending", "rejected")
                    else o
                )
                for o in self.transfer_offers
            ),
        )
        return candidate._record_history("選手退団") if record else candidate

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
        return next(
            (team for team in self.opponent_teams if name in team.members), None
        )

    def roster_owner(self, name):
        # Presets may reuse club members; affiliation is owned_players vs rivals.
        return None

    def with_opponent_teams(self, teams):
        self = remember_loyalties(self)
        if any(not run.completed for run in self.tournaments):
            raise SeasonSaveError(
                "大会参加中は他チームの所属設定を変更できません。大会終了後に取り込んでください。"
            )
        teams = tuple(teams)
        ids = {team.id for team in teams}
        offers = tuple(
            (
                replace(o, status="cancelled")
                if o.status in ("pending", "rejected") and o.team_id not in ids
                else o
            )
            for o in self.transfer_offers
        )
        candidate = replace(
            self, opponent_teams=teams, transfer_offers=offers
        ).with_registered_ratings()
        candidate.validate()
        return candidate._record_history("他チーム設定更新")

    def with_new_team(self):
        used_names = {team.name for team in self.teams}
        number = 1
        while f"編成{number}" in used_names:
            number += 1
        return replace(
            self,
            preset_name=f"編成{number}",
            roster=(),
            editing_team_id=None,
            preset_igl=None,
            preset_carrier=None,
            preset_ai=DEFAULT_TEAM_AI,
        )

    def with_editing_team(self, team_id):
        team = self.team(team_id)
        if team is None:
            raise SeasonSaveError("編集するチームが見つかりません。")
        return replace(
            self,
            preset_name=team.name,
            roster=team.roster,
            editing_team_id=team.id,
            preset_igl=team.igl,
            preset_carrier=team.carrier,
            preset_ai=team.ai,
        )

    def with_confirmed_team(self):
        if self.unplayable_roster:
            raise SeasonSaveError(
                f"{'、'.join(self.unplayable_roster)}: {EXPIRED_ROSTER_WARNING}。契約状況で再契約してください。"
            )
        if not self.roster_ready:
            raise SeasonSaveError("チームの登録には5人の編成が必要です。")
        name = self.preset_name.strip()
        if any(
            team.name.casefold() == name.casefold() and team.id != self.editing_team_id
            for team in self.teams
        ):
            raise SeasonSaveError(
                "同じ名前の編成プリセットが登録されています。別のプリセット名を入力してください。"
            )
        team = SeasonTeam(
            self.editing_team_id or uuid4().hex,
            name,
            self.roster,
            self.preset_igl,
            self.preset_carrier,
            self.preset_ai,
        )
        teams = tuple(team if old.id == team.id else old for old in self.teams)
        if self.editing_team_id is None:
            teams += (team,)
        candidate = replace(
            self, teams=teams, editing_team_id=team.id
        ).with_registered_ratings()
        candidate.validate()
        return candidate

    def with_selected_team(self, team_id):
        candidate = replace(self, selected_team_id=team_id)
        candidate.validate()
        return candidate

    def with_roster(self, names):
        roster = tuple(names)
        expired = tuple(
            name
            for name in roster
            if self.player(name) is not None and not self.can_play(name)
        )
        if expired:
            raise SeasonSaveError(
                f"{'、'.join(expired)}: {EXPIRED_ROSTER_WARNING}。契約状況で再契約してください。"
            )
        candidate = replace(
            self,
            roster=roster,
            preset_igl=self.preset_igl if self.preset_igl in roster else None,
            preset_carrier=(
                self.preset_carrier if self.preset_carrier in roster else None
            ),
        )
        candidate.validate()
        return candidate

    def with_preset_settings(self, *, igl=None, carrier=None, ai=DEFAULT_TEAM_AI):
        candidate = replace(self, preset_igl=igl, preset_carrier=carrier, preset_ai=ai)
        editing = self.team(self.editing_team_id)
        # An unchanged saved lineup can receive settings immediately. A new or
        # edited roster stays a draft until the five-player confirmation.
        if editing is not None and editing.roster == self.roster:
            candidate = replace(
                candidate,
                teams=tuple(
                    (
                        replace(t, igl=igl, carrier=carrier, ai=ai)
                        if t.id == editing.id
                        else t
                    )
                    for t in self.teams
                ),
            )
        candidate.validate()
        return candidate

    def with_team_name(self, name):
        candidate = replace(self, team_name=name.strip()).with_registered_ratings()
        candidate.validate()
        return candidate._record_history("チーム名変更")

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
                raise SeasonSaveError(
                    f"{name}は「{owner.name}」に所属しているため、所持選手に追加できません。"
                )
            if name in existing:
                raise SeasonSaveError(
                    f"{name}はすでに所持しています（または入力が重複しています）。"
                )
            player = next((p for p in self.scout_players if p.name == name), None)
            if player is None:
                raise SeasonSaveError(
                    f"選手 {name} が見つかりません。既存のプレイヤー名を正確に入力してください。"
                )
            player = self.salary_player(player)
            if self.contract_refused(name):
                raise SeasonSaveError(f"{name}はこのチームとの契約を永久に拒否しています。")
            players.append(player)
            contracts = [c for c in contracts if c.player_name != name]
            contracts.append(
                replace(
                    initial_contract(player, self.game_month, signed_on=self.date.isoformat()),
                    team_loyalty=signing_loyalty(self, name, self.club_id,
                                               "short" if player.loyalty == 0 else "year1")[0],
                    signing_number=signing_loyalty(self, name, self.club_id,
                                                  "short" if player.loyalty == 0 else "year1")[1],
                )
            )
            existing.add(name)
        candidate = replace(
            self, owned_players=tuple(players), contracts=tuple(contracts)
        )
        candidate.validate()
        return candidate._record_history("選手追加")

    def without_player(self, name):
        if name in self.roster:
            raise SeasonSaveError("先にその選手をロスターから外してください。")
        if self.player_in_saved_team(name):
            raise SeasonSaveError(
                "先に登録済みチームの編成からその選手を外し、編成を確定してください。"
            )
        candidate = replace(
            self,
            owned_players=tuple(p for p in self.owned_players if p.name != name),
            contracts=tuple(
                replace(c, end_reason="released") if c.player_name == name else c
                for c in self.contracts
            ),
            transfer_offers=tuple(
                (
                    replace(o, status="cancelled")
                    if o.player_name == name and o.status in ("pending", "rejected")
                    else o
                )
                for o in self.transfer_offers
            ),
        )
        candidate.validate()
        return candidate._record_history("選手放出")

    def validate(self):
        try:
            validate_loyalties(self.team_loyalties)
            validate_contract_memory(self)
        except (ValueError, TypeError) as exc:
            raise SeasonSaveError(str(exc)) from exc
        if type(self.day_advance_pending) is not bool:
            raise SeasonSaveError("アクションの日付進行待ちデータが不正です。")
        try:
            validate_uses(self.scout_uses, self.date)
        except (ValueError, TypeError) as exc:
            raise SeasonSaveError(f"スカウト実績が不正です: {exc}") from exc
        try:
            validate_pair_settings()
            validate_pair_days(self.pair_days)
            validate_news(self.pair_news, self.date)
            if type(self.pair_familiarity_seed) is not int:
                raise ValueError("ペア練度のシードは整数にしてください。")
        except (ValueError, TypeError, IndexError) as exc:
            raise SeasonSaveError(f"ペア練度データが不正です: {exc}") from exc
        if (
            type(self.transfer_multiplier) not in (int, float)
            or not math.isfinite(self.transfer_multiplier)
            or self.transfer_multiplier < 0
            or type(rival_settings.NON_REGULAR_OFFER_CHANCE) not in (int, float)
            or not 0 <= rival_settings.NON_REGULAR_OFFER_CHANCE <= 1
            or type(rival_settings.OFFER_SURPLUS_MONTHS) is not int
            or rival_settings.OFFER_SURPLUS_MONTHS < 0
        ):
            raise SeasonSaveError(
                "ライバル経済設定の移籍金倍率・オファー確率・給与余裕月数が不正です。"
            )
        if not isinstance(self.history, tuple):
            raise SeasonSaveError("シーズン履歴の形式が不正です。")
        if not isinstance(self.salary_mode, SalaryMode):
            raise SeasonSaveError("給与モードが不正です。")
        if not isinstance(self.salary_records, tuple):
            raise SeasonSaveError("給与一覧の形式が不正です。")
        if self.salary_mode == SalaryMode.STATIC:
            if (
                self.salary_settings is not None
                or self.salary_records
                or self.salary_updated_month is not None
            ):
                raise SeasonSaveError("静的月給モードの設定が不正です。")
        else:
            if (
                not isinstance(self.salary_settings, SalarySettings)
                or not self.salary_records
            ):
                raise SeasonSaveError("成績連動月給の設定または給与一覧がありません。")
            try:
                self.salary_settings.validate()
                names = set()
                for record in self.salary_records:
                    if not isinstance(record, SalaryRecord) or record.name in names:
                        raise SalaryDataError(
                            "給与一覧の選手が不正または重複しています。"
                        )
                    record.validate()
                    names.add(record.name)
            except SalaryDataError as exc:
                raise SeasonSaveError(str(exc)) from exc
            if (
                type(self.salary_updated_month) is not int
                or self.salary_updated_month != self.game_month
            ):
                raise SeasonSaveError("給与更新月とゲーム内の経過月数が一致しません。")
        if type(self.game_month) is not int or self.game_month < 0:
            raise SeasonSaveError("ゲーム内の経過月数が不正です。")
        if (
            type(self.monthly_events_through) is not int
            or not 0 <= self.monthly_events_through <= self.game_month
        ):
            raise SeasonSaveError("月次イベントの処理済み月が不正です。")
        event_ids = set()
        last_date = parse_date(self.start_date)
        if not isinstance(self.monthly_events, tuple):
            raise SeasonSaveError("月次イベント履歴の形式が不正です。")
        for event in self.monthly_events:
            if (
                not isinstance(event, MonthlyEvent)
                or not isinstance(event.id, str)
                or not event.id
                or event.id in event_ids
                or not isinstance(event.kind, str)
                or event.kind
                not in {
                    "recruitment",
                    "recruitment_unfilled",
                    "renewal",
                    "renewal_waiting",
                    "departure",
                    "month_completed",
                    "roster_return",
                    "transfer_offer",
                }
                or event.team_id is not None
                and (not isinstance(event.team_id, str) or not event.team_id)
                or not isinstance(event.team_name, str)
                or not event.team_name
                or event.player_name is not None
                and (not isinstance(event.player_name, str) or not event.player_name)
                or not isinstance(event.message, str)
                or not event.message
            ):
                raise SeasonSaveError("月次イベント履歴が不正です。")
            try:
                date = parse_date(event.date)
            except CompetitionError as exc:
                raise SeasonSaveError("月次イベントの日付が不正です。") from exc
            # Daily expiries and tournament settlements can precede the first
            # monthly settlement now that contracts expire on arbitrary days.
            daily_contract_event = event.id.startswith("contract-end:") or ":after:" in event.id
            if (
                date < last_date
                or date > self.date
                or not (0 if daily_contract_event else 1)
                <= month_index(parse_date(self.start_date), date)
                <= (self.game_month if daily_contract_event else self.monthly_events_through)
            ):
                raise SeasonSaveError("月次イベントの日付と処理済み月が一致しません。")
            event_ids.add(event.id)
            last_date = date
        if type(self.sponsor_active) is not bool:
            raise SeasonSaveError("スポンサー契約の状態が不正です。")
        if not isinstance(self.transfer_offers, tuple):
            raise SeasonSaveError("移籍オファーの形式が不正です。")
        offer_ids, pending_names = set(), set()
        club_ids = {
            club.id
            for club in self.opponent_teams
            if isinstance(club, SeasonClub) and isinstance(club.id, str)
        }
        for offer in self.transfer_offers:
            if (
                not isinstance(offer, TransferOffer)
                or not isinstance(offer.id, str)
                or not offer.id
                or offer.id in offer_ids
                or not isinstance(offer.team_id, str)
                or not offer.team_id
                or not isinstance(offer.player_name, str)
                or not offer.player_name
                or type(offer.created_month) is not int
                or not 0 <= offer.created_month <= self.game_month
                or type(offer.fee) is not int
                or offer.fee < 0
                or offer.status
                not in ("pending", "accepted", "rejected", "forced", "cancelled")
            ):
                raise SeasonSaveError("移籍オファーが不正です。")
            offer_ids.add(offer.id)
            terms_fields = (
                offer.contract_kind,
                offer.contract_months,
                offer.monthly_salary,
            )
            if any(v is not None for v in terms_fields):
                if (
                    offer.contract_kind not in CONTRACT_OPTIONS.values()
                    or type(offer.contract_months) is not int
                    or type(offer.monthly_salary) is not int
                    or offer.monthly_salary < 0
                    or offer.contract_kind == "short"
                    and not 1 <= offer.contract_months <= 6
                    or offer.contract_kind != "short"
                    and offer.contract_months
                    != {"year1": 12, "year2": 24, "year3": 36}.get(offer.contract_kind)
                ):
                    raise SeasonSaveError("移籍オファーの契約条件が不正です。")
            if offer.status == "pending":
                if (
                    offer.player_name in pending_names
                    or self.player(offer.player_name) is None
                    or offer.team_id not in club_ids
                ):
                    raise SeasonSaveError(
                        "保留中の移籍オファーと所持選手が一致しません。"
                    )
                pending_names.add(offer.player_name)
        if (
            not isinstance(self.transferred_players, tuple)
            or any(not isinstance(n, str) or not n for n in self.transferred_players)
            or len(set(self.transferred_players)) != len(self.transferred_players)
        ):
            raise SeasonSaveError("移籍履歴が不正です。")
        if not isinstance(self.developed_players, tuple):
            raise SeasonSaveError("育成済み選手の形式が不正です。")
        developed_names = set()
        for player in self.developed_players:
            validate_player(player)
            if player.name in developed_names:
                raise SeasonSaveError("育成済み選手の記録が重複または不正です。")
            developed_names.add(player.name)
        if self.world_level_lock is not None:
            try:
                validate_world_level(self.world_level_lock)
            except WorldLevelError as exc:
                raise SeasonSaveError(
                    f"大会中の世界レベル設定が不正です: {exc}"
                ) from exc
        if (
            not isinstance(self.rated_results, tuple)
            or any(not isinstance(n, str) or not n for n in self.rated_results)
            or len(set(self.rated_results)) != len(self.rated_results)
        ):
            raise SeasonSaveError("レート更新済み試合の記録が不正です。")
        rating_ids = set()
        for rating in self.ratings:
            if (
                not isinstance(rating, SeasonRating)
                or not isinstance(rating.team_id, str)
                or not rating.team_id
                or rating.team_id in rating_ids
                or not isinstance(rating.team_name, str)
                or not rating.team_name.strip()
                or type(rating.value) not in (int, float)
                or not math.isfinite(rating.value)
                or rating.value < 0
            ):
                raise SeasonSaveError("レーティング一覧が不正です。")
            rating_ids.add(rating.team_id)
        if type(self.money) is not int:
            raise SeasonSaveError("所持金は円単位の整数で保存してください。")
        if type(self.game_month) is not int or self.game_month < 0:
            raise SeasonSaveError("ゲーム内の経過月数が不正です。")
        try:
            if (
                self.date < parse_date(self.start_date)
                or month_index(parse_date(self.start_date), self.date)
                != self.game_month
            ):
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
        if len(set(pool_names)) != len(pool_names):
            raise SeasonSaveError("初期キャラ候補が重複しています。")
        if (
            any(not isinstance(name, str) for name in self.starter_selection)
            or len(self.starter_selection) > ROSTER_SIZE
            or len(set(self.starter_selection)) != len(self.starter_selection)
            or not set(self.starter_selection).issubset(pool_names)
        ):
            raise SeasonSaveError(
                "初期キャラの選択が不正です。候補から異なる5人まで選んでください。"
            )
        if self.starter_selection_pending:
            if len(pool_names) < ROSTER_SIZE:
                raise SeasonSaveError(
                    "INITIAL_OWNED_PLAYERS に初期キャラ候補を5人以上設定してください。"
                )
            if (
                self.owned_players
                or self.roster
                or self.teams
                or self.contracts
                or self.tournaments
                or self.game_month != 0
                or self.date != parse_date(self.start_date)
                or self.money != INITIAL_MONEY
            ):
                raise SeasonSaveError(
                    "初期キャラの5人を確定してからゲームを開始してください。"
                )
        elif self.starter_candidates and len(self.starter_selection) != ROSTER_SIZE:
            raise SeasonSaveError("入手済みの初期キャラの選択記録には5人が必要です。")
        if not isinstance(self.team_name, str) or not self.team_name.strip():
            raise SeasonSaveError("チーム名を入力してください。")
        if len(self.team_name) > 40:
            raise SeasonSaveError("チーム名は40文字以内にしてください。")
        if not isinstance(self.club_id, str) or not self.club_id:
            raise SeasonSaveError("自チームのIDが不正です。")
        if (
            not isinstance(self.preset_name, str)
            or not self.preset_name.strip()
            or len(self.preset_name) > 40
        ):
            raise SeasonSaveError("編成プリセット名は1～40文字で入力してください。")
        names = []
        for player in self.owned_players:
            validate_player(player)
            names.append(player.name)
        if len(set(names)) != len(names):
            raise SeasonSaveError("所持選手が重複しています。")
        contract_names = set()
        for contract in self.contracts:
            if (
                not isinstance(contract, PlayerContract)
                or not isinstance(contract.player_name, str)
                or not contract.player_name
            ):
                raise SeasonSaveError("選手の契約データが不正です。")
            if contract.player_name in contract_names:
                raise SeasonSaveError("選手の契約が重複しています。")
            contract_names.add(contract.player_name)
            if (
                type(contract.start_month) is not int
                or not 0 <= contract.start_month <= self.game_month
                or type(contract.duration_months) is not int
                or type(contract.monthly_salary) is not int
                or contract.monthly_salary < 0
            ):
                raise SeasonSaveError("契約期間または月給が不正です。")
            if (
                not isinstance(contract.kind, str)
                or contract.kind not in CONTRACT_OPTIONS.values()
                or (contract.kind == "short" and not 1 <= contract.duration_months <= 6)
                or (
                    contract.kind != "short"
                    and contract.duration_months
                    != {"year1": 12, "year2": 24, "year3": 36}.get(contract.kind)
                )
            ):
                raise SeasonSaveError("契約の種類と期間が一致しません。")
            if type(contract.team_loyalty) not in (int, float) or not math.isfinite(
                contract.team_loyalty
            ):
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
        validate_preset_settings(
            self.roster, self.preset_igl, self.preset_carrier, self.preset_ai
        )
        team_ids = set()
        team_names = set()
        affiliations = {}
        for team in self.teams:
            if (
                not isinstance(team, SeasonTeam)
                or not isinstance(team.id, str)
                or not team.id
            ):
                raise SeasonSaveError("登録済みチームのIDが不正です。")
            if (
                not isinstance(team.name, str)
                or not team.name.strip()
                or len(team.name) > 40
            ):
                raise SeasonSaveError("登録済みチームの名前が不正です。")
            if team.id in team_ids or team.name.casefold() in team_names:
                raise SeasonSaveError("登録済みチームが重複しています。")
            if len(team.roster) != ROSTER_SIZE or any(
                not isinstance(name, str) for name in team.roster
            ):
                raise SeasonSaveError("登録済みチームは5人の編成が必要です。")
            if len(set(team.roster)) != ROSTER_SIZE or not set(team.roster).issubset(
                names
            ):
                raise SeasonSaveError(
                    "登録済みチームに重複または未所持の選手が含まれています。"
                )
            if team.id == self.club_id:
                raise SeasonSaveError("編成プリセットと自チームのIDが重複しています。")
            validate_preset_settings(team.roster, team.igl, team.carrier, team.ai)
            team_ids.add(team.id)
            team_names.add(team.name.casefold())

        for team_id in (self.editing_team_id, self.selected_team_id):
            if team_id is not None and (
                not isinstance(team_id, str) or team_id not in team_ids
            ):
                raise SeasonSaveError("編集・使用チームの指定が不正です。")
        team_names = {self.team_name.casefold()}
        team_ids.add(self.club_id)
        owned_names = set(names)
        regular_names = set()
        for team in self.opponent_teams:
            if (
                not isinstance(team, SeasonClub)
                or not isinstance(team.id, str)
                or not team.id
            ):
                raise SeasonSaveError("シーズン所属チームのIDが不正です。")
            if (
                not isinstance(team.name, str)
                or not team.name.strip()
                or len(team.name) > 40
            ):
                raise SeasonSaveError(
                    "シーズン所属チーム名は1～40文字で設定してください。"
                )
            if team.id in team_ids or team.name.casefold() in team_names:
                raise SeasonSaveError(
                    f"シーズン所属チーム「{team.name}」が重複しています。"
                )
            if (
                type(team.transfer_multiplier) not in (int, float)
                or not math.isfinite(team.transfer_multiplier)
                or team.transfer_multiplier < 0
            ):
                raise SeasonSaveError(
                    f"「{team.name}」の移籍金倍率は0以上の有限の数値で設定してください。"
                )
            if type(team.money) is not int or type(team.sponsor_active) is not bool:
                raise SeasonSaveError(
                    f"「{team.name}」の資金またはスポンサー契約が不正です。"
                )
            validate_initial_rating(team.initial_rating, team.name)
            if (
                not isinstance(team.acquired_members, tuple)
                or any(not isinstance(n, str) for n in team.acquired_members)
                or len(set(team.acquired_members)) != len(team.acquired_members)
                or not set(team.acquired_members).issubset(team.members)
            ):
                raise SeasonSaveError(f"「{team.name}」の獲得選手の記録が不正です。")
            if team.world_level_lock is not None:
                try:
                    validate_world_level(team.world_level_lock)
                except WorldLevelError as exc:
                    raise SeasonSaveError(
                        f"「{team.name}」の世界レベル固定が不正です: {exc}"
                    ) from exc
            if (
                not isinstance(team.regular_members, tuple)
                or any(not isinstance(n, str) or not n for n in team.regular_members)
                or len(set(team.regular_members)) != len(team.regular_members)
                or regular_names.intersection(team.regular_members)
            ):
                raise SeasonSaveError(f"「{team.name}」の正規メンバーが不正です。")
            for role in (team.regular_igl, team.regular_carrier):
                if role is not None and (
                    not isinstance(role, str)
                    or role not in team.regular_members[:ROSTER_SIZE]
                ):
                    raise SeasonSaveError(
                        f"「{team.name}」の正規メンバーの役割が不正です。"
                    )
            regular_names.update(team.regular_members)
            if (
                not isinstance(team.initial_shared_members, tuple)
                or any(
                    not isinstance(n, str) or not n for n in team.initial_shared_members
                )
                or len(set(team.initial_shared_members))
                != len(team.initial_shared_members)
            ):
                raise SeasonSaveError(
                    f"「{team.name}」の初期所属抽選の記録が不正です。"
                )
            if (
                not isinstance(team.contracts, tuple)
                or not isinstance(team.preferred_roles, tuple)
                or len(team.preferred_roles) > ROSTER_SIZE
                or any(
                    not isinstance(role, str) or not role
                    for role in team.preferred_roles
                )
            ):
                raise SeasonSaveError(f"「{team.name}」のロール構成が不正です。")
            contract_names = set()
            for contract in team.contracts:
                if (
                    not isinstance(contract, PlayerContract)
                    or not isinstance(contract.player_name, str)
                    or not contract.player_name
                    or contract.player_name in contract_names
                    or type(contract.start_month) is not int
                    or not 0 <= contract.start_month <= self.game_month
                    or type(contract.duration_months) is not int
                    or type(contract.monthly_salary) is not int
                    or contract.monthly_salary < 0
                    or contract.kind not in CONTRACT_OPTIONS.values()
                    or contract.kind == "short"
                    and not 1 <= contract.duration_months <= 6
                    or contract.kind != "short"
                    and contract.duration_months
                    != {"year1": 12, "year2": 24, "year3": 36}.get(contract.kind)
                    or type(contract.team_loyalty) not in (int, float)
                    or not math.isfinite(contract.team_loyalty)
                    or contract.end_reason not in (None, "left", "released")
                    or (contract.end_reason is None)
                    != (contract.player_name in team.members)
                ):
                    raise SeasonSaveError(f"「{team.name}」の契約データが不正です。")
                contract_names.add(contract.player_name)
            for player in team.players:
                validate_player(player)
                name = player.name
                if name in owned_names:
                    raise SeasonSaveError(
                        f"{name}は自分の所持選手と「{team.name}」に重複所属しています。"
                    )
                if name in affiliations:
                    raise SeasonSaveError(
                        f"{name}が「{affiliations[name]}」と「{team.name}」に重複所属しています。"
                    )
                affiliations[name] = team.name
            for label, value in (("IGL", team.igl), ("キャリアー", team.carrier)):
                if value is not None and (
                    not isinstance(value, str) or value not in team.roster
                ):
                    raise SeasonSaveError(
                        f"「{team.name}」の{label}は、先頭5人の出場選手から指定してください。"
                    )
            from roster_select import TEAM_AI_OPTIONS

            if not isinstance(team.ai, str) or team.ai not in TEAM_AI_OPTIONS.values():
                raise SeasonSaveError(f"「{team.name}」のAI設定が不正です: {team.ai!r}")
            team_ids.add(team.id)
            team_names.add(team.name.casefold())

        try:
            self._validate_competitions()
        except (CompetitionError, TypeError, AttributeError) as exc:
            raise SeasonSaveError(f"大会データが不正です: {exc}") from exc
        self.world_level_settings

    def _validate_competitions(self):
        from roster_select import TEAM_AI_OPTIONS

        seen = set()
        for run in self.tournaments:
            if not isinstance(run, TournamentProgress) or run.tournament_id in seen:
                raise CompetitionError("大会進行データが重複または不正です。")
            seen.add(run.tournament_id)
            event = self.tournament_definition(run.tournament_id)
            if (
                event is None
                or type(run.completed) is not bool
                or type(run.declined) is not bool
            ):
                raise CompetitionError("大会設定または進行状態が不正です。")
            if (
                type(run.prize_paid) is not int
                or run.prize_paid < 0
                or type(run.seed) is not int
                or not 0 <= run.seed < 2**31
            ):
                raise CompetitionError("大会賞金または乱数シードが不正です。")
            if bool(run.results) != (run.last_match_date is not None):
                raise CompetitionError("大会の最終試合日が結果と一致しません。")
            if run.last_match_date is not None:
                if (
                    not parse_date(event.start_date)
                    <= parse_date(run.last_match_date)
                    <= self.date
                ):
                    raise CompetitionError("大会の最終試合日が不正です。")
            if run.completed_date != (
                run.last_match_date if run.completed and not run.declined else None
            ):
                raise CompetitionError("大会の終了日が完了状態と一致しません。")
            if run.declined:
                if (
                    not event.participation_optional
                    or not event.allow_player_entry
                    or not run.completed
                    or run.entrants
                    or run.results
                    or run.ranking
                    or run.prize_paid
                    or run.own_team_id is not None
                ):
                    raise CompetitionError("不参加の大会データが不正です。")
                continue
            if not 2 <= len(run.entrants) <= event.team_count:
                raise CompetitionError("大会の参加チーム数が設定と一致しません。")
            ids, names, player_names = set(), set(), set()
            for team in run.entrants:
                if (
                    not isinstance(team, CompetitionTeam)
                    or not isinstance(team.id, str)
                    or not team.id
                    or team.id in ids
                ):
                    raise CompetitionError("大会の参加チームIDが重複または不正です。")
                if (
                    not isinstance(team.name, str)
                    or not team.name.strip()
                    or team.name.casefold() in names
                ):
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
                if (
                    team.ai not in TEAM_AI_OPTIONS.values()
                    or team.igl not in members
                    or team.carrier not in members
                ):
                    raise CompetitionError(
                        "大会チームのAI・IGL・キャリアーが不正です。"
                    )
            if (run.own_team_id is not None and not event.allow_player_entry) or (
                run.own_team_id is not None and run.own_team_id not in ids
            ):
                raise CompetitionError("プレイヤーの参加状態が不正です。")
            if (
                run.own_team_id is not None
                and (
                    run.own_team_id != self.club_id
                    or not isinstance(run.preset_id, str)
                    or not run.preset_id
                )
                or run.own_team_id is None
                and run.preset_id is not None
            ):
                raise CompetitionError(
                    "大会の自チーム・編成プリセットの指定が不正です。"
                )
            pending, ranking = next_match(event, run)
            for score in run.results:
                if score.decided_by_rating and run.own_team_id in (
                    score.left_id,
                    score.right_id,
                ):
                    raise CompetitionError(
                        "レート判定は他チーム同士の試合にのみ使用できます。"
                    )
            if run.completed != (pending is None) or run.ranking != ranking:
                raise CompetitionError(
                    "大会の完了状態または順位が試合結果と一致しません。"
                )
            expected = (
                event.prizes.get(ranking.index(run.own_team_id) + 1, 0)
                if ranking and run.own_team_id
                else 0
            )
            if run.prize_paid != expected:
                raise CompetitionError("大会順位と受け取り賞金が一致しません。")
            if not run.completed and run.own_team_id:
                team = self.team(run.preset_id)
                snapshot = next(t for t in run.entrants if t.id == run.own_team_id)
                departed = any(self.player(p.name) is None for p in snapshot.players)
                if (team is None and not departed) or (
                    team is not None
                    and team.roster != tuple(p.name for p in snapshot.players)
                ):
                    raise CompetitionError(
                        "大会参加中のチームの編成は変更できません。大会終了後に変更してください。"
                    )


def configured_season_teams(
    existing_teams=(),
    transferred_players=(),
    *,
    game_month=0,
    start_date=None,
    game_date=None,
    monthly_events=(),
    priority_names=(),
    loyalty_memory=None,
    contract_memory=None,
    refusal_memory=(),
):
    from realtime_season_teams import SEASON_TEAMS

    start_date = start_date or configured_calendar()[0]
    game_date = game_date or add_months(parse_date(start_date), game_month).isoformat()

    if not isinstance(SEASON_TEAMS, (list, tuple)):
        raise SeasonSaveError("SEASON_TEAMS はチーム設定のリストにしてください。")
    saved_players = {p.name: p for team in existing_teams for p in team.players}
    saved_teams = {team.name: team for team in existing_teams}
    automatic_signings = {
        (event.team_id, event.player_name)
        for event in monthly_events
        if event.kind == "recruitment"
    }
    automatic_departures = {
        (event.team_id, event.player_name)
        for event in monthly_events
        if event.kind == "departure"
    }

    def excluded(team_name, player_name):
        old = saved_teams.get(team_name) if isinstance(team_name, str) else None
        identifier = "world-" + uuid5(NAMESPACE_URL, "realtime-season:" + team_name.strip()).hex if isinstance(team_name, str) else None
        return (
            player_name in transferred_players
            or (player_name, identifier) in refusal_memory
            or old is not None
            and (old.id, player_name) in automatic_departures
            and player_name not in old.members
        )

    configured_names = {
        p
        for item in SEASON_TEAMS
        if isinstance(item, dict) and isinstance(item.get("players"), (list, tuple))
        for p in item["players"]
        if isinstance(p, str) and not excluded(item.get("name"), p)
    }
    teams = []
    for item in SEASON_TEAMS:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"].strip()
        ):
            raise SeasonSaveError(
                "各シーズンチームに name（チーム名）を設定してください。"
            )
        name = item["name"].strip()
        club_id = "world-" + uuid5(NAMESPACE_URL, "realtime-season:" + name).hex
        saved_team = saved_teams.get(name)
        members = item.get("players")
        if not isinstance(members, (list, tuple)) or any(
            not isinstance(p, str) or not p.strip() for p in members
        ):
            raise SeasonSaveError(
                f"「{name}」の players に所属選手名のリストを設定してください。"
            )
        if len(set(members)) != len(members):
            raise SeasonSaveError(
                f"「{name}」の所属選手名は、同じチーム内で重複させないでください。"
            )
        players = []
        for member in members:
            if excluded(name, member):
                continue
            player = saved_players.get(member) or get_by_name(member)
            if player is None:
                raise SeasonSaveError(f"「{name}」の選手 {member} が見つかりません。")
            players.append(player)
        if saved_team and (saved_team.regular_members or saved_team.acquired_members):
            # Import metadata without resetting an evolving season lineup.
            players = list(saved_team.players)
        elif saved_team:
            players.extend(
                p
                for p in saved_team.players
                if (saved_team.id, p.name) in automatic_signings
                and p.name not in configured_names
                and p.name not in transferred_players
            )
        contracts = (
            {c.player_name: c for c in saved_team.contracts} if saved_team else {}
        )
        members_now = {p.name for p in players}
        contracts = {
            name: (
                c
                if name in members_now
                else replace(c, end_reason=c.end_reason or "released")
            )
            for name, c in contracts.items()
        }
        for player in players:
            if (
                player.name not in contracts
                or contracts[player.name].end_reason is not None
            ):
                count = (contract_memory or {}).get(player.name, {}).get(club_id, 0)
                kind = "short" if player.loyalty == 0 else "year1"
                loyalty = ((loyalty_memory or {}).get(player.name, {}).get(club_id, 50.0)
                           + CONTRACT_LOYALTY[kind][1] if count else CONTRACT_LOYALTY[kind][0])
                contracts[player.name] = replace(
                    initial_contract(player, game_month, signed_on=game_date), team_loyalty=loyalty, signing_number=count + 1
                )
        roster = {p.name for p in players[:ROSTER_SIZE]}
        igl, carrier = item.get("igl"), item.get("carrier")
        if (
            igl in transferred_players
            or igl in members[:ROSTER_SIZE]
            and igl not in roster
        ):
            igl = None
        if (
            carrier in transferred_players
            or carrier in members[:ROSTER_SIZE]
            and carrier not in roster
        ):
            carrier = None
        teams.append(
            SeasonClub(
                club_id,
                name,
                tuple(players),
                igl=igl,
                carrier=carrier,
                ai=item.get("ai", DEFAULT_TEAM_AI),
                transfer_multiplier=item.get("transfer_multiplier", 12.0),
                preferred_roles=(
                    saved_team.preferred_roles
                    if saved_team and saved_team.preferred_roles
                    else tuple(p.role for p in players[:ROSTER_SIZE])
                ),
                contracts=tuple(contracts.values()),
                regular_members=saved_team.regular_members if saved_team else (),
                regular_igl=saved_team.regular_igl if saved_team else None,
                regular_carrier=saved_team.regular_carrier if saved_team else None,
                money=(
                    saved_team.money
                    if saved_team
                    else item.get("initial_money", rival_settings.INITIAL_MONEY)
                ),
                sponsor_active=(
                    saved_team.sponsor_active
                    if saved_team
                    else item.get("sponsor_active", True)
                ),
                acquired_members=saved_team.acquired_members if saved_team else (),
                world_level_lock=saved_team.world_level_lock if saved_team else None,
                initial_rating=item.get(
                    "initial_rating",
                    saved_team.initial_rating if saved_team else DEFAULT_TEAM_RATING,
                ),
                initial_shared_members=(
                    saved_team.initial_shared_members if saved_team else ()
                ),
            )
        )
    # Validate individual definitions before resolving overlap, so invalid
    # settings are still rejected even when a player loses an allocation.
    for team in teams:
        SeasonState(
            "マイチーム", (), opponent_teams=(team,), game_month=game_month,
            start_date=start_date, game_date=game_date,
        ).validate()
    teams = resolve_club_memberships(
        teams, priority_names, existing_teams=existing_teams
    )
    # Check team names, membership, reserves, and ability snapshots as one batch.
    SeasonState(
        "マイチーム", (), opponent_teams=tuple(teams), game_month=game_month,
        start_date=start_date, game_date=game_date,
    ).validate()
    return tuple(teams)


def new_season(starter_names=None, *, salary_mode=SalaryMode.STATIC):
    from realtime_season_config import INITIAL_TEAM_RATING

    validate_initial_rating(INITIAL_TEAM_RATING, "マイチーム")
    try:
        validate_pair_settings()
    except (ValueError, TypeError, IndexError) as exc:
        raise SeasonSaveError(f"ペア練度設定が不正です: {exc}") from exc
    choose_starters = starter_names is None
    if starter_names is None:
        from realtime_season_config import INITIAL_OWNED_PLAYERS

        starter_names = INITIAL_OWNED_PLAYERS
    if not isinstance(starter_names, (list, tuple)) or any(
        not isinstance(name, str) or not name.strip() for name in starter_names
    ):
        raise SeasonSaveError(
            "初期所持プレイヤーは、プレイヤー名の文字列を並べたリストで設定してください。"
        )
    try:
        start, periods, events = configured_calendar()
    except CompetitionError as exc:
        raise SeasonSaveError(str(exc)) from exc
    state = SeasonState(
        "マイチーム",
        (),
        opponent_teams=configured_season_teams(
            priority_names=starter_names if not choose_starters else (),
            start_date=start, game_date=start,
        ),
        start_date=start,
        game_date=start,
        in_season_periods=periods,
        tournament_definitions=events,
        ratings=(SeasonRating(PLAYER_CLUB_ID, "マイチーム", INITIAL_TEAM_RATING),),
        pair_familiarity_seed=pair_settings.INITIAL_SEED,
    ).with_registered_ratings()
    if not choose_starters:
        state = initial_pair_days(state)
    state = state._record_history("シーズン開始")
    if not choose_starters:
        # Explicit names remain available for administrative setup and simulations.
        state = state.with_added_players(starter_names)
        if SalaryMode(salary_mode) == SalaryMode.KD_DYNAMIC:
            state = replace(
                state,
                salary_mode=SalaryMode.KD_DYNAMIC,
                salary_settings=configured_salary_settings(),
            ).with_updated_salaries(initial=True)
        return state._record_history("初期設定完了")
    candidates = []
    for name in starter_names:
        player = get_by_name(name)
        if player is None:
            raise SeasonSaveError(f"初期キャラ候補 {name} が見つかりません。")
        candidates.append(player)
    state = replace(
        state, starter_candidates=tuple(candidates), starter_selection_pending=True
    )
    # Validate the entire configured pool before drawing, including entries
    # that might otherwise be hidden by the random selection.
    state.validate()
    if len(candidates) > INITIAL_CANDIDATE_COUNT:
        sampled = {p.name for p in Random().sample(candidates, INITIAL_CANDIDATE_COUNT)}
        state = replace(
            state, starter_candidates=tuple(p for p in candidates if p.name in sampled)
        )
    return state.with_salary_mode(salary_mode)._record_history("初期設定完了")


class SeasonStore:
    def __init__(self, path=DEFAULT_SAVE_PATH):
        self.path = Path(path).resolve()
        self.history_path = self.path.with_name(f"{self.path.stem}_history.json")
        self.history_export_error = None

    def _export_history(self, state):
        # The save is authoritative. An export failure must not trigger a replay
        # of an already committed paid action; the next load/save rebuilds it.
        try:
            export_history(self.history_path, state)
            self.history_export_error = None
        except (OSError, TypeError, ValueError) as exc:
            self.history_export_error = str(exc)

    def import_season_teams(self, state):
        state = remember_loyalties(state)
        candidate = state.with_opponent_teams(
            configured_season_teams(
                state.opponent_teams,
                state.transferred_players,
                game_month=state.game_month,
                start_date=state.start_date,
                game_date=state.date.isoformat(),
                monthly_events=state.monthly_events,
                priority_names=tuple(p.name for p in state.owned_players),
                loyalty_memory=state.team_loyalties,
                contract_memory=state.contract_signings,
                refusal_memory=state.contract_bans,
            )
        )
        self.save(candidate)
        return candidate

    def import_competitions(self, state):
        try:
            protected = {run.tournament_id for run in state.tournaments}
            # Started and completed events retain their rules and prize snapshots.
            events = tuple(
                event for event in state.tournament_definitions if event.id in protected
            )
            _, periods, definitions = configured_calendar(on_date=state.date, existing_definitions=events)
            events += tuple(event for event in definitions if event.id not in protected)
            candidate = replace(
                state, in_season_periods=periods, tournament_definitions=events
            )
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
            raise SeasonSaveError(
                "セーブデータはUTF-8形式で保存してください。"
            ) from exc
        try:
            data = json.loads(contents)
            if not isinstance(data, dict):
                raise SeasonSaveError(
                    "セーブデータはJSONオブジェクトである必要があります。"
                )
            if type(data.get("version")) is not int or data["version"] not in range(
                1, SAVE_VERSION + 1
            ):
                raise SeasonSaveError(
                    "このセーブデータのバージョンには対応していません。"
                )

            def load_player(row):
                if data["version"] >= 16 and (
                    not isinstance(row, dict)
                    or not {"research_level", "aim_lab_level"}.issubset(row)
                ):
                    raise SeasonSaveError(
                        "選手の育成レベルがセーブデータにありません。"
                    )
                return player_from_save(row, legacy_loyalty=data["version"] < 7)

            if not isinstance(data.get("owned_players"), list) or not isinstance(
                data.get("roster"), list
            ):
                raise SeasonSaveError("所持選手またはロスターの形式が不正です。")
            teams = ()
            editing_team_id = None
            selected_team_id = None
            if data["version"] == 1 and len(data["roster"]) == ROSTER_SIZE:
                teams = (
                    SeasonTeam("legacy-team", data["team_name"], tuple(data["roster"])),
                )
                editing_team_id = "legacy-team"
            elif data["version"] >= 2:
                if not isinstance(data.get("teams"), list):
                    raise SeasonSaveError("登録済みチームの形式が不正です。")
                parsed_teams = []
                for item in data["teams"]:
                    if not isinstance(item, dict) or not isinstance(
                        item.get("roster"), list
                    ):
                        raise SeasonSaveError("登録済みチームの形式が不正です。")
                    parsed_teams.append(
                        SeasonTeam(
                            item["id"],
                            item["name"],
                            tuple(item["roster"]),
                            item.get("igl"),
                            item.get("carrier"),
                            item.get("ai", DEFAULT_TEAM_AI),
                        )
                    )
                teams = tuple(parsed_teams)
                editing_team_id = data.get("editing_team_id")
                selected_team_id = data.get("selected_team_id")
            opponent_teams = ()
            if data["version"] >= 3:
                if not isinstance(data.get("opponent_teams"), list):
                    raise SeasonSaveError("シーズン所属チームの形式が不正です。")
                regular_config = {}
                if data["version"] < 19:
                    from realtime_season_teams import SEASON_TEAMS

                    if isinstance(SEASON_TEAMS, (list, tuple)):
                        regular_config = {
                            row["name"]: row
                            for row in SEASON_TEAMS
                            if isinstance(row, dict)
                            and isinstance(row.get("name"), str)
                        }
                clubs = []
                for item in data["opponent_teams"]:
                    if not isinstance(item, dict) or not isinstance(
                        item.get("players"), list
                    ):
                        raise SeasonSaveError("シーズン所属チームの形式が不正です。")
                    if data["version"] >= 9 and (
                        not isinstance(item.get("contracts"), list)
                        or not isinstance(item.get("preferred_roles"), list)
                    ):
                        raise SeasonSaveError(
                            "他チームの契約またはロール構成の形式が不正です。"
                        )
                    if data["version"] >= 15 and not isinstance(
                        item.get("regular_members"), list
                    ):
                        raise SeasonSaveError("正規メンバーの形式が不正です。")
                    if data["version"] >= 19 and (
                        "money" not in item
                        or "sponsor_active" not in item
                        or not isinstance(item.get("acquired_members"), list)
                    ):
                        raise SeasonSaveError("ライバルチームの資金データが不正です。")
                    regular = item.get("regular_members", ())
                    if data["version"] >= 20 and "initial_rating" not in item:
                        raise SeasonSaveError(
                            "ライバルチームの初期レートがありません。"
                        )
                    if data["version"] >= 23 and not isinstance(
                        item.get("initial_shared_members"), list
                    ):
                        raise SeasonSaveError("初期所属抽選の記録がありません。")
                    regular_igl, regular_carrier = item.get("regular_igl"), item.get(
                        "regular_carrier"
                    )
                    if (
                        data["version"] == 14
                        and "regular_members" not in item
                        and item["name"] in regular_config
                    ):
                        original = regular_config[item["name"]]
                        members = original.get("players")
                        if (
                            isinstance(members, (list, tuple))
                            and len(members) >= ROSTER_SIZE
                            and all(
                                isinstance(n, str) and get_by_name(n) is not None
                                for n in members
                            )
                            and len(set(members)) == len(members)
                        ):
                            regular = members
                            regular_igl = (
                                original.get("igl")
                                if original.get("igl") in members[:ROSTER_SIZE]
                                else None
                            )
                            regular_carrier = (
                                original.get("carrier")
                                if original.get("carrier") in members[:ROSTER_SIZE]
                                else None
                            )
                    clubs.append(
                        SeasonClub(
                            item["id"],
                            item["name"],
                            tuple(load_player(p) for p in item["players"]),
                            igl=item.get("igl"),
                            carrier=item.get("carrier"),
                            ai=item.get("ai", DEFAULT_TEAM_AI),
                            transfer_multiplier=item.get("transfer_multiplier", 12.0),
                            preferred_roles=tuple(item.get("preferred_roles", ())),
                            contracts=tuple(
                                PlayerContract(**c) for c in item.get("contracts", ())
                            ),
                            regular_members=tuple(regular),
                            regular_igl=regular_igl,
                            regular_carrier=regular_carrier,
                            money=item.get(
                                "money",
                                regular_config.get(item["name"], {}).get(
                                    "initial_money", rival_settings.INITIAL_MONEY
                                ),
                            ),
                            sponsor_active=item.get("sponsor_active", True),
                            initial_rating=item.get(
                                "initial_rating", DEFAULT_TEAM_RATING
                            ),
                            initial_shared_members=tuple(
                                item.get("initial_shared_members", ())
                            ),
                            acquired_members=tuple(item.get("acquired_members", ())),
                            world_level_lock=(
                                WorldLevel(**item["world_level_lock"])
                                if item.get("world_level_lock") is not None
                                else None
                            ),
                        )
                    )
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
                if not isinstance(
                    data.get("tournament_definitions"), list
                ) or not isinstance(data.get("tournaments"), list):
                    raise SeasonSaveError("大会設定または進行データの形式が不正です。")
                events = tuple(
                    definition_from_dict(row) for row in data["tournament_definitions"]
                )
                runs = []
                for row in data["tournaments"]:
                    if not isinstance(row, dict):
                        raise SeasonSaveError("大会進行データの形式が不正です。")
                    values = dict(row)
                    entrants = []
                    for team in values["entrants"]:
                        team = dict(team)
                        team.setdefault("ai", DEFAULT_TEAM_AI)
                        team["players"] = tuple(load_player(p) for p in team["players"])
                        entrants.append(CompetitionTeam(**team))
                    values["entrants"] = tuple(entrants)
                    values["results"] = tuple(
                        SeriesScore(**result) for result in values["results"]
                    )
                    values["ranking"] = tuple(values["ranking"])
                    if data["version"] < 10:
                        # Older saves have no match dates. Keep every result and
                        # resume on the next day without rewriting the saved file.
                        values.setdefault(
                            "last_match_date", game_date if values["results"] else None
                        )
                        values.setdefault(
                            "completed_date",
                            (
                                values["last_match_date"]
                                if values["completed"] and not values["declined"]
                                else None
                            ),
                        )
                    runs.append(TournamentProgress(**values))
                runs = tuple(runs)
            else:
                start_date, periods, events = configured_calendar()
                game_date = add_months(parse_date(start_date), game_month).isoformat()
                runs = ()
            if data["version"] >= 6:
                if not isinstance(
                    data.get("starter_candidates"), list
                ) or not isinstance(data.get("starter_selection"), list):
                    raise SeasonSaveError("初期キャラ候補または選択の形式が不正です。")
                starter_candidates = tuple(
                    load_player(p) for p in data["starter_candidates"]
                )
                starter_selection = tuple(data["starter_selection"])
                starter_selection_pending = data["starter_selection_pending"]
            else:
                # Existing saves keep their inventory; never discard players to restart selection.
                starter_candidates, starter_selection, starter_selection_pending = (
                    (),
                    (),
                    False,
                )
            ratings, rated_results, sponsor_active, transferred_players = (
                (),
                (),
                True,
                (),
            )
            if data["version"] >= 8:
                if any(
                    not isinstance(data.get(key), list)
                    for key in ("ratings", "rated_results", "transferred_players")
                ):
                    raise SeasonSaveError(
                        "レーティングまたは移籍履歴の形式が不正です。"
                    )
                ratings = tuple(SeasonRating(**row) for row in data["ratings"])
                rated_results = tuple(data["rated_results"])
                sponsor_active = data["sponsor_active"]
                transferred_players = tuple(data["transferred_players"])
            monthly_events, monthly_events_through = (), game_month
            transfer_offers = ()
            if data["version"] >= 15:
                if not isinstance(data.get("transfer_offers"), list):
                    raise SeasonSaveError("移籍オファーの形式が不正です。")
                transfer_offers = tuple(
                    TransferOffer(**row) for row in data["transfer_offers"]
                )
            if data["version"] >= 9:
                if not isinstance(data.get("monthly_events"), list):
                    raise SeasonSaveError("月次イベント履歴の形式が不正です。")
                monthly_events = tuple(
                    MonthlyEvent(**row) for row in data["monthly_events"]
                )
                monthly_events_through = data["monthly_events_through"]
            opponent_teams = tuple(
                replace(
                    club,
                    preferred_roles=club.preferred_roles
                    or tuple(p.role for p in club.players[:ROSTER_SIZE]),
                    contracts=club.contracts
                    or tuple(initial_contract(p, game_month) for p in club.players),
                )
                for club in opponent_teams
            )
            salary_mode, salary_settings, salary_rows, salary_month = (
                SalaryMode.STATIC,
                None,
                (),
                None,
            )
            developed_players = ()
            history = ()
            if data["version"] >= 18:
                if not isinstance(data.get("history"), list):
                    raise SeasonSaveError("シーズン履歴の形式が不正です。")
                history = tuple(data["history"])
            world_level_lock = None
            if data["version"] >= 17:
                locked = data["world_level_lock"]
                if locked is not None:
                    if not isinstance(locked, dict):
                        raise SeasonSaveError(
                            "大会中の世界レベル設定の形式が不正です。"
                        )
                    world_level_lock = WorldLevel(**locked)
            if data["version"] >= 16:
                if not isinstance(data.get("developed_players"), list):
                    raise SeasonSaveError("育成済み選手の形式が不正です。")
                developed_players = tuple(
                    load_player(p) for p in data["developed_players"]
                )
            if data["version"] >= 14:
                salary_mode = SalaryMode(data["salary_mode"])
                settings = data["salary_settings"]
                if settings is not None:
                    settings = dict(settings)
                    settings["fixed_salaries"] = tuple(
                        tuple(row) for row in settings["fixed_salaries"]
                    )
                    settings["excluded_files"] = tuple(
                        settings.get("excluded_files", ())
                    )
                    salary_settings = SalarySettings(**settings)
                if not isinstance(data["salary_records"], list):
                    raise SeasonSaveError("給与一覧の形式が不正です。")
                salary_rows = tuple(
                    SalaryRecord(**row) for row in data["salary_records"]
                )
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
                transfer_offers=transfer_offers,
                preset_name=(
                    data["preset_name"]
                    if data["version"] >= 11
                    else data.get("preset_name", data["team_name"])
                ),
                club_id=(
                    data["club_id"]
                    if data["version"] >= 11
                    else data.get("club_id", PLAYER_CLUB_ID)
                ),
                preset_igl=data.get("preset_igl"),
                preset_carrier=data.get("preset_carrier"),
                preset_ai=data.get("preset_ai", DEFAULT_TEAM_AI),
                salary_mode=salary_mode,
                salary_settings=salary_settings,
                salary_records=salary_rows,
                salary_updated_month=salary_month,
                developed_players=developed_players,
                world_level_lock=world_level_lock,
                history=history,
                pair_days=decode_pair_days(data.get("pair_days", [])),
                pair_familiarity_seed=data.get(
                    "pair_familiarity_seed", pair_settings.INITIAL_SEED
                ),
                pair_news=tuple(
                    PairNews(**{**row, "pair": tuple(row["pair"])})
                    for row in data.get("pair_news", [])
                ),
                scout_uses=tuple(tuple(row) for row in data.get("scout_uses", [])),
                day_advance_pending=data.get("day_advance_pending", False),
                team_loyalties=data.get("team_loyalties", {}),
                contract_signings=data.get("contract_signings", {}),
                contract_bans=tuple(tuple(pair) for pair in data.get("contract_bans", [])),
            )
            old_layout = data["version"] < 11 and "club_id" not in data
            required_ids = {t.id for t in state.opponent_teams}
            if old_layout:
                required_ids.update(t.id for t in state.teams)
            else:
                required_ids.add(state.club_id)
            if data["version"] >= 8 and not required_ids.issubset(
                {r.team_id for r in state.ratings}
            ):
                raise SeasonSaveError(
                    "所属チームのレーティングがセーブデータにありません。"
                )
            if old_layout:
                old_ids = {t.id for t in state.teams} | {
                    r.own_team_id for r in state.tournaments if r.own_team_id
                }
                active = next(
                    (r for r in state.tournaments if not r.completed and r.own_team_id),
                    None,
                )
                source_id = (
                    active.own_team_id
                    if active
                    else state.selected_team_id
                    or (state.teams[0].id if state.teams else None)
                )
                source = state.team(source_id)
                source = source or next(
                    (
                        t
                        for r in state.tournaments
                        for t in r.entrants
                        if t.id == source_id
                    ),
                    None,
                )
                actual_name = source.name if source else state.team_name
                own_rating = next(
                    (r.value for r in state.ratings if r.team_id == source_id),
                    DEFAULT_TEAM_RATING,
                )
                migrated_runs = []

                def club_key(key):
                    return state.club_id if key in old_ids else key

                for run in state.tournaments:
                    migrated_runs.append(
                        replace(
                            run,
                            own_team_id=(
                                club_key(run.own_team_id) if run.own_team_id else None
                            ),
                            preset_id=run.own_team_id,
                            entrants=tuple(
                                (
                                    replace(t, id=club_key(t.id), name=actual_name)
                                    if t.id in old_ids
                                    else t
                                )
                                for t in run.entrants
                            ),
                            results=tuple(
                                replace(
                                    s,
                                    left_id=club_key(s.left_id),
                                    right_id=club_key(s.right_id),
                                )
                                for s in run.results
                            ),
                            ranking=tuple(club_key(key) for key in run.ranking),
                        )
                    )
                state = replace(
                    state,
                    team_name=actual_name,
                    tournaments=tuple(migrated_runs),
                    ratings=(
                        *tuple(
                            r
                            for r in state.ratings
                            if r.team_id not in old_ids and r.team_id != state.club_id
                        ),
                        SeasonRating(state.club_id, actual_name, own_rating),
                    ),
                )
            state = state.with_registered_ratings()
            if data["version"] < 17:
                state = state._with_tournament_world_level()
            elif bool(state.active_tournaments) != (state.world_level_lock is not None):
                raise SeasonSaveError(
                    "大会の進行状態と固定された世界レベルが一致しません。"
                )
            if data["version"] < 19:
                # Start the rival economy now; never replay past income or prizes.
                from season_rival_economy import signing_terms, transfer_cost

                offers = []
                for offer in state.transfer_offers:
                    player = state.player(offer.player_name)
                    if player is not None and offer.status in ("pending", "rejected"):
                        terms = signing_terms(state, player)
                        offer = replace(
                            offer,
                            fee=transfer_cost(player, state.transfer_multiplier),
                            contract_kind=terms.kind,
                            contract_months=terms.months,
                            monthly_salary=terms.monthly_salary,
                        )
                    offers.append(offer)
                state = replace(
                    state, transfer_offers=tuple(offers)
                )._with_tournament_world_level()
            if data["version"] >= 21 and (
                "pair_days" not in data
                or "pair_familiarity_seed" not in data
                or not isinstance(data.get("pair_news"), list)
            ):
                raise SeasonSaveError("ペア練度データがありません。")
            if "pair_days" not in data and not state.starter_selection_pending:
                state = initial_pair_days(state)
            if data["version"] >= 22 and not isinstance(data.get("scout_uses"), list):
                raise SeasonSaveError("スカウト実績データがありません。")
            if (
                data["version"] >= 24
                and type(data.get("day_advance_pending")) is not bool
            ):
                raise SeasonSaveError("アクションの日付進行待ちデータが不正です。")
            if data["version"] >= 25 and not isinstance(
                data.get("team_loyalties"), dict
            ):
                raise SeasonSaveError("チーム別忠誠データがありません。")
            if data["version"] >= 26 and (not isinstance(data.get("contract_signings"), dict)
                                         or not isinstance(data.get("contract_bans"), list)):
                raise SeasonSaveError("契約回数・再契約拒否データがありません。")
            state.validate()
            if "team_loyalties" not in data or "contract_signings" not in data:
                state = remember_loyalties(state)
            validate_history(state.history)
            if not state.history:
                state = state._record_history("記録開始（旧セーブ）")
            self._export_history(state)
            return state
        except (KeyError, TypeError, ValueError) as exc:
            raise SeasonSaveError(f"セーブデータを読み込めません: {exc}") from exc

    def save(self, state):
        state.validate()
        try:
            validate_history(state.history)
        except (TypeError, ValueError) as exc:
            raise SeasonSaveError(str(exc)) from exc
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
            "salary_settings": (
                asdict(state.salary_settings)
                if state.salary_settings is not None
                else None
            ),
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
            "tournament_definitions": [
                asdict(event) for event in state.tournament_definitions
            ],
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
            "transfer_offers": [asdict(offer) for offer in state.transfer_offers],
            "developed_players": [asdict(player) for player in state.developed_players],
            "world_level_lock": (
                asdict(state.world_level_lock)
                if state.world_level_lock is not None
                else None
            ),
            "history": list(state.history),
            "pair_days": encode_pair_days(state.pair_days),
            "pair_familiarity_seed": state.pair_familiarity_seed,
            "pair_news": [asdict(e) for e in state.pair_news],
            "scout_uses": state.scout_uses,
            "day_advance_pending": state.day_advance_pending,
            "team_loyalties": state.team_loyalties,
            "contract_signings": state.contract_signings,
            "contract_bans": state.contract_bans,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
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
        self._export_history(state)
