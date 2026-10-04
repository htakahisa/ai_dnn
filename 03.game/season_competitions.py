"""Calendar, validated configuration, and resumable pure tournament brackets."""

from calendar import monthrange
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
import re

from character_stats import CharacterStats


class CompetitionError(ValueError):
    pass


def parse_date(value):
    try:
        if not isinstance(value, str):
            raise ValueError()
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError()
        return parsed
    except ValueError as exc:
        raise CompetitionError(f"日付はYYYY-MM-DDで指定してください: {value!r}") from exc


def parse_month_day(value):
    if not isinstance(value, str) or re.fullmatch(r"\d{2}-\d{2}", value) is None:
        raise CompetitionError("毎年の開催日・告知日はMM-DDで指定してください。")
    return parse_date("2000-" + value)


def annual_date(year, month_day, *, notice=False):
    if type(year) is not int or not 1 <= year <= 9999:
        raise CompetitionError("毎年開催の大会日付を計算できません。開催年・告知日を確認してください。")
    value = parse_month_day(month_day)
    try:
        return date(year, value.month, value.day)
    except ValueError:
        if (value.month, value.day) != (2, 29):
            raise
        # There is no February 29 tournament in a common year; a notice
        # scheduled for that date appears on February 28 instead.
        return date(year, 2, 28) if notice else None


def add_months(day, months):
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(day.day, monthrange(year, month + 1)[1]))


def month_index(start, current):
    return (current.year - start.year) * 12 + current.month - start.month


def validate_periods(periods):
    if not isinstance(periods, (tuple, list)):
        raise CompetitionError("IN_SEASON_PERIODS は月日の組のリストにしてください。")
    for pair in periods:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise CompetitionError("インシーズン期間は ('03-01', '11-30') のように設定してください。")
        for value in pair:
            parse_date("2000-" + value if isinstance(value, str) else value)


def phase_for(day, periods):
    md = day.strftime("%m-%d")
    for first, last in periods:
        if (first <= md <= last if first <= last else md >= first or md <= last):
            return "in_season"
    return "off_season"


@dataclass(frozen=True)
class TournamentDefinition:
    id: str
    name: str
    start_date: str
    team_count: int
    prizes: dict[int, int]
    normal_maps_to_win: int = 2
    lower_final_maps_to_win: int = 3
    grand_final_maps_to_win: int = 3
    format: str = "double_elimination"
    appearance_conditions: dict = field(default_factory=dict)
    participation_optional: bool = True
    allow_player_entry: bool = True
    opponent_teams: tuple[str, ...] = ()
    visible_from: str = "2026-01-01"
    annual_id: str | None = None
    annual_notice: str | None = None

    @property
    def display_name(self):
        return f"{self.name} {parse_date(self.start_date).year}" if self.annual_id else self.name

    def for_year(self, year):
        if self.annual_id is None:
            return self
        original = parse_date(self.start_date)
        start = annual_date(year, original.strftime("%m-%d"))
        if start is None:
            return None
        notice = parse_date(self.visible_from)
        notice_year = year + notice.year - original.year
        visible = annual_date(notice_year, self.annual_notice or notice.strftime("%m-%d"), notice=True)
        return replace(self, id=f"{self.annual_id}_{year}", start_date=start.isoformat(), visible_from=visible.isoformat())

    @property
    def match_count(self):
        return 2 * self.team_count - 2 if self.format == "double_elimination" else self.team_count - 1

    @property
    def end_date(self):
        try:
            return (parse_date(self.start_date) + timedelta(days=self.match_count - 1)).isoformat()
        except (OverflowError, TypeError) as exc:
            raise CompetitionError("大会の終了予定日を計算できません。開始日・参加数を確認してください。") from exc

    def validate(self):
        for value in (self.id, self.name):
            if not isinstance(value, str) or not value.strip() or len(value) > 80:
                raise CompetitionError("大会のID・名前は1～80文字で設定してください。")
        if re.fullmatch(r"[A-Za-z0-9_-]+", self.id) is None:
            raise CompetitionError("大会IDは英数字・ハイフン・アンダースコアで指定してください。")
        parse_date(self.start_date)
        if self.annual_id is not None:
            if (not isinstance(self.annual_id, str)
                    or re.fullmatch(r"[A-Za-z0-9_-]+", self.annual_id) is None
                    or self.id != f"{self.annual_id}_{parse_date(self.start_date).year}"):
                raise CompetitionError("毎年開催の大会IDが不正です。")
            if self.annual_notice is not None:
                parse_month_day(self.annual_notice)
                notice = parse_date(self.visible_from)
                if annual_date(notice.year, self.annual_notice, notice=True) != notice:
                    raise CompetitionError("毎年開催の告知月日と告知日が一致しません。")
        elif self.annual_notice is not None:
            raise CompetitionError("毎年開催ではない大会に告知月日が指定されています。")
        if parse_date(self.visible_from) > parse_date(self.start_date):
            raise CompetitionError(f"{self.name}: 告知開始日は開催開始日以前にしてください。")
        if self.format not in ("double_elimination", "single_elimination"):
            raise CompetitionError("大会形式が不正です。")
        minimum = 4 if self.format == "double_elimination" else 2
        if type(self.team_count) is not int or not minimum <= self.team_count <= 32:
            raise CompetitionError(f"{self.name}: 参加チーム数は{minimum}～32で設定してください。")
        parse_date(self.end_date)
        for value in (self.normal_maps_to_win, self.lower_final_maps_to_win, self.grand_final_maps_to_win):
            if type(value) is not int or not 1 <= value <= 10:
                raise CompetitionError("先取マップ数は1～10の整数で設定してください。")
        if type(self.participation_optional) is not bool or type(self.allow_player_entry) is not bool:
            raise CompetitionError("参加設定はTrueまたはFalseで設定してください。")
        if not isinstance(self.prizes, dict) or any(type(rank) is not int or not 1 <= rank <= self.team_count
                or type(money) is not int or money < 0 for rank, money in self.prizes.items()):
            raise CompetitionError("賞金は参加数以内の順位と0以上の円単位整数で設定してください。")
        if (not isinstance(self.opponent_teams, (tuple, list))
                or any(not isinstance(name, str) or not name.strip() for name in self.opponent_teams)
                or len(set(self.opponent_teams)) != len(self.opponent_teams)):
            raise CompetitionError("opponent_teams は重複のないチーム名リストにしてください。")
        c = self.appearance_conditions
        if not isinstance(c, dict) or set(c) - {"min_money", "min_owned_players", "completed_tournaments", "best_rank", "phase"}:
            raise CompetitionError("大会の出現条件に未対応の項目があります。")
        for key in ("min_money", "min_owned_players"):
            if key in c and (type(c[key]) is not int or c[key] < 0):
                raise CompetitionError(f"{key} は0以上の整数で設定してください。")
        required = c.get("completed_tournaments", [])
        if not isinstance(required, (list, tuple)) or any(not isinstance(item, str) for item in required):
            raise CompetitionError("completed_tournaments は大会IDのリストにしてください。")
        ranks = c.get("best_rank", {})
        if not isinstance(ranks, dict) or any(not isinstance(key, str) or type(rank) is not int or rank < 1 for key, rank in ranks.items()):
            raise CompetitionError("best_rank は大会IDと必要順位の辞書にしてください。")
        if "phase" in c and c["phase"] not in ("in_season", "off_season"):
            raise CompetitionError("phase はin_seasonまたはoff_seasonです。")

    def appears(self, state):
        if state.date < parse_date(self.visible_from):
            return False
        c = self.appearance_conditions
        if ("min_money" in c and state.money < c["min_money"]) or len(state.owned_players) < c.get("min_owned_players", 0):
            return False
        if c.get("phase", state.phase) != state.phase:
            return False
        for event_id in c.get("completed_tournaments", []):
            run = self.condition_run(state, event_id)
            if run is None or not run.completed or run.own_team_id is None:
                return False
        for event_id, rank in c.get("best_rank", {}).items():
            run = self.condition_run(state, event_id)
            if run is None or not run.completed or run.own_team_id not in run.ranking or run.ranking.index(run.own_team_id) + 1 > rank:
                return False
        return True

    def condition_run(self, state, event_id):
        year = parse_date(self.start_date).year
        annual = any(event.annual_id == event_id for event in state.tournament_definitions)
        run = state.tournament(f"{event_id}_{year}" if annual else event_id)
        if run is None and annual:
            # An imported old one-off may use the bare ID instead of ID_year.
            legacy = state.tournament_definition(event_id)
            if legacy is not None and parse_date(legacy.start_date).year == year:
                run = state.tournament(event_id)
        return run


@dataclass(frozen=True)
class CompetitionTeam:
    id: str
    name: str
    players: tuple[CharacterStats, ...]
    ai: str
    igl: str
    carrier: str


@dataclass(frozen=True)
class SeriesScore:
    match_id: str
    left_id: str
    right_id: str
    left_wins: int
    right_wins: int
    decided_by_rating: bool = False


@dataclass(frozen=True)
class TournamentProgress:
    tournament_id: str
    own_team_id: str | None
    entrants: tuple[CompetitionTeam, ...] = ()
    results: tuple[SeriesScore, ...] = ()
    ranking: tuple[str, ...] = ()
    completed: bool = False
    declined: bool = False
    prize_paid: int = 0
    seed: int = 0
    last_match_date: str | None = None
    completed_date: str | None = None
    preset_id: str | None = None


@dataclass(frozen=True)
class BracketMatch:
    id: str
    stage: str
    left: str
    right: str
    maps_to_win: int


@dataclass(frozen=True)
class BracketSlot:
    team_id: str | None
    source_match: str | None = None
    source_outcome: str | None = None


@dataclass(frozen=True)
class BracketViewMatch:
    match: BracketMatch
    left: BracketSlot
    right: BracketSlot
    score: SeriesScore | None = None
    pending: bool = False


def player_eliminated(definition, progress):
    if progress.own_team_id is None or progress.declined:
        return False
    losses = sum(1 for score in progress.results
                 if progress.own_team_id in (score.left_id, score.right_id)
                 and (score.left_id if score.left_wins > score.right_wins else score.right_id) != progress.own_team_id)
    return losses >= (2 if definition.format == "double_elimination" else 1)


def bracket_view(definition, progress):
    """Project every card and feeder using the same bracket as real matches.

    Provisional winners only advance the generator; unresolved slots display
    their source card instead of inventing a team or a predicted winner.
    """
    if progress.declined or not progress.entrants:
        return ()
    pending, _ = next_match(definition, progress)
    scores = {s.match_id: s for s in progress.results}
    origins = {}
    cards = []

    def slot(team_id):
        source, outcome = origins.get(team_id, (None, None))
        return BracketSlot(team_id if source is None or source in scores else None, source, outcome)

    generator = bracket(definition, progress.entrants)
    try:
        match = next(generator)
        while True:
            score = scores.get(match.id)
            cards.append(BracketViewMatch(match, slot(match.left), slot(match.right), score,
                                          pending is not None and match.id == pending.id))
            winner = (score.left_id if score.left_wins > score.right_wins else score.right_id) if score else match.left
            loser = match.right if winner == match.left else match.left
            origins[winner] = (match.id, "winner")
            origins[loser] = (match.id, "loser")
            match = generator.send(winner)
    except StopIteration:
        return tuple(cards)


def bracket(definition, entrants):
    """Send each series winner to resume a bracket, including non-power-of-two byes."""
    ids = [team.id for team in entrants]
    size = 2
    seeds = [1, 2]
    while size < len(ids):
        size *= 2
        seeds = [value for seed in seeds for value in (seed, size + 1 - seed)]
    winners = [ids[rank - 1] if rank <= len(ids) else None for rank in seeds]
    eliminations = []

    def match(match_id, stage, left, right):
        if left is None or right is None:
            return left or right, None
        need = {"lower_final": definition.lower_final_maps_to_win,
                "grand_final": definition.grand_final_maps_to_win}.get(stage, definition.normal_maps_to_win)
        winner = yield BracketMatch(match_id, stage, left, right, need)
        if winner not in (left, right):
            raise CompetitionError("勝者が対戦チームに含まれていません。")
        return winner, right if winner == left else left

    lower = []
    wr, lr = 1, 0
    while len(winners) > 1:
        next_winners, incoming = [], []
        for index in range(0, len(winners), 2):
            stage = "grand_final" if definition.format == "single_elimination" and len(winners) == 2 else "upper"
            winner, loser = yield from match(f"W{wr}M{index // 2 + 1}", stage, winners[index], winners[index + 1])
            next_winners.append(winner)
            if loser:
                incoming.append(loser)
        if definition.format == "single_elimination":
            eliminations.extend(incoming)
            winners, wr = next_winners, wr + 1
            continue
        incoming.reverse()
        if wr == 1:
            lr += 1
            if len(incoming) % 2:
                lower.append(incoming.pop(0))
            for index in range(0, len(incoming), 2):
                winner, loser = yield from match(f"L{lr}M{index // 2 + 1}", "lower", incoming[index], incoming[index + 1])
                lower.append(winner)
                eliminations.append(loser)
        else:
            for group, target in ((lower, incoming), (incoming, lower)):
                while len(group) > len(target):
                    lr += 1
                    count = min(len(group) // 2, len(group) - len(target))
                    if count == 0:
                        raise CompetitionError("Lowerブラケットの人数調整に失敗しました。")
                    carry = len(group) - count * 2
                    reduced = group[:carry]
                    for index in range(carry, len(group), 2):
                        winner, loser = yield from match(f"L{lr}M{(index - carry) // 2 + 1}", "lower", group[index], group[index + 1])
                        reduced.append(winner)
                        eliminations.append(loser)
                    group[:] = reduced
            lr += 1
            merged = []
            for index, (survivor, dropped) in enumerate(zip(lower, incoming), 1):
                final = len(next_winners) == 1 and len(lower) == 1
                winner, loser = yield from match("LOWER_FINAL" if final else f"L{lr}M{index}",
                                                "lower_final" if final else "lower", survivor, dropped)
                merged.append(winner)
                eliminations.append(loser)
            lower = merged
        winners, wr = next_winners, wr + 1
    if definition.format == "single_elimination":
        return (winners[0], *reversed(eliminations))
    winner, loser = yield from match("GRAND_FINAL", "grand_final", winners[0], lower[0])
    return (winner, loser, *reversed(eliminations))


def next_match(definition, progress):
    """Rebuild from persisted series results; no generator state is stored in JSON."""
    if progress.declined:
        return None, ()
    generator = bracket(definition, progress.entrants)
    consumed = 0
    try:
        pending = next(generator)
        for result in progress.results:
            if type(result.decided_by_rating) is not bool:
                raise CompetitionError("レート判定の記録が不正です。")
            if (result.match_id, result.left_id, result.right_id) != (pending.id, pending.left, pending.right):
                raise CompetitionError("保存された大会試合の順序または対戦チームが不正です。")
            a, b, need = result.left_wins, result.right_wins, pending.maps_to_win
            if type(a) is not int or type(b) is not int or not ((a == need and 0 <= b < need) or (b == need and 0 <= a < need)):
                raise CompetitionError("シリーズのマップ勝利数が先取マップ数と一致しません。")
            consumed += 1
            pending = generator.send(pending.left if a > b else pending.right)
        return pending, ()
    except StopIteration as finished:
        if consumed != len(progress.results) or consumed == 0:
            raise CompetitionError("大会の試合結果が不足しています。")
        return None, tuple(finished.value)


def definition_from_dict(row, *, year=None):
    if not isinstance(row, dict):
        raise CompetitionError("大会設定は辞書で指定してください。")
    row = dict(row)
    if isinstance(row.get("start_date"), str) and len(row["start_date"]) == 5:
        md = row["start_date"]
        parse_month_day(md)
        if "annual_id" in row or "annual_notice" in row:
            raise CompetitionError("MM-DDの設定に内部の年度データは指定できません。")
        base_id = row.get("id")
        if not isinstance(base_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", base_id) is None:
            raise CompetitionError("大会IDは英数字・ハイフン・アンダースコアで指定してください。")
        if year is None:
            from realtime_season_competitions import START_DATE
            year = parse_date(START_DATE).year
        start = annual_date(year, md)
        while start is None:
            year += 1
            if year > 9999:
                raise CompetitionError("次回の開催日を計算できません。開催年を確認してください。")
            start = annual_date(year, md)
        notice = row.get("visible_from", "01-01")
        if isinstance(notice, str) and len(notice) == 5:
            visible = annual_date(year - (notice > md), notice, notice=True)
            notice_md = notice
        else:
            visible = parse_date(notice)
            notice_md = visible.strftime("%m-%d")
        row.update(id=f"{base_id}_{year}", start_date=start.isoformat(),
                   visible_from=visible.isoformat(), annual_id=base_id, annual_notice=notice_md)
    # Old configuration and saved rules may still contain a manual end date.
    # Duration is now derived from the actual bracket's series count.
    row.pop("end_date", None)
    if isinstance(row.get("prizes"), dict):
        restored = {}
        for key, value in row["prizes"].items():
            # JSON object keys are strings; accept only canonical integer strings.
            if isinstance(key, str) and key.isdecimal() and str(int(key)) == key:
                key = int(key)
            restored[key] = value
        row["prizes"] = restored
    opponents = row.get("opponent_teams", ())
    if not isinstance(opponents, (list, tuple)):
        raise CompetitionError("opponent_teams はチーム名のリストにしてください。")
    row["opponent_teams"] = tuple(opponents)
    try:
        definition = TournamentDefinition(**row)
        definition.validate()
        return definition
    except TypeError as exc:
        raise CompetitionError(f"大会設定の項目が不正です: {exc}") from exc


def configured_calendar(*, on_date=None, existing_definitions=()):
    from realtime_season_competitions import START_DATE, IN_SEASON_PERIODS, TOURNAMENTS
    parse_date(START_DATE)
    validate_periods(IN_SEASON_PERIODS)
    if not isinstance(TOURNAMENTS, (list, tuple)):
        raise CompetitionError("TOURNAMENTS は大会設定のリストにしてください。")
    day = parse_date(START_DATE) if on_date is None else on_date
    definitions = []
    for row in TOURNAMENTS:
        annual = isinstance(row, dict) and isinstance(row.get("start_date"), str) and len(row["start_date"]) == 5
        default_notice = "01-01" if annual else START_DATE
        definitions.append(definition_from_dict({"visible_from": default_notice, **row}
                                               if isinstance(row, dict) else row, year=day.year))
    definitions = tuple(definitions)
    ids = [event.annual_id or event.id for event in definitions]
    if len(set(ids)) != len(ids):
        raise CompetitionError("大会IDが重複しています。")
    definitions = extend_annual_calendar(definitions, day, day, omit_past=True)
    validation = {event.id: event for event in (*definitions, *existing_definitions)}
    validate_definitions(tuple(validation.values()))
    return START_DATE, tuple(tuple(pair) for pair in IN_SEASON_PERIODS), definitions


def extend_annual_calendar(definitions, day, first_day, *, omit_past=False):
    """Persist annual occurrences ahead of time without changing old results."""
    sources = {}
    for definition in definitions:
        if definition.annual_id is not None:
            sources[definition.annual_id] = definition
    events = [event for event in definitions
              if not (omit_past and event.annual_id and parse_date(event.start_date) < first_day)]
    ids = {event.id for event in events}
    # A February 29 source can be several years ahead; generate its same-year
    # qualifiers as well so dependency validation remains complete.
    last_year = max([day.year + 1, *(parse_date(source.start_date).year for source in sources.values())])
    for year in range(day.year, min(last_year, 9999) + 1):
        for source in sources.values():
            event = source.for_year(year)
            if event is None or parse_date(event.start_date) < first_day or event.id in ids:
                continue
            events.append(event)
            ids.add(event.id)
    return tuple(events)


def validate_definitions(definitions):
    seen = set()
    for definition in definitions:
        if not isinstance(definition, TournamentDefinition):
            raise CompetitionError("大会設定の形式が不正です。")
        definition.validate()
        if definition.id in seen:
            raise CompetitionError(f"大会IDが重複しています: {definition.id}")
        seen.add(definition.id)
    aliases = {d.annual_id for d in definitions if d.annual_id is not None}
    valid_ids = seen | aliases
    for definition in definitions:
        dependencies = set(definition.appearance_conditions.get("completed_tournaments", [])) | set(definition.appearance_conditions.get("best_rank", {}))
        if definition.id in dependencies or definition.annual_id in dependencies or dependencies - valid_ids:
            raise CompetitionError(f"{definition.name}: 出現条件の大会IDが不正です。")
    by_id = {d.id: d for d in definitions}
    graph = {d.id: set() for d in definitions}
    for d in definitions:
        dependencies = set(d.appearance_conditions.get("completed_tournaments", [])) | set(d.appearance_conditions.get("best_rank", {}))
        for key in dependencies:
            target = key
            if key in aliases:
                year = parse_date(d.start_date).year
                target = f"{key}_{year}"
                if target not in seen:
                    legacy = by_id.get(key)
                    target = key if legacy and parse_date(legacy.start_date).year == year else None
            if target is not None:
                graph[d.id].add(target)
    def visit(node, path):
        if node in path:
            raise CompetitionError("大会の出現条件が循環しています。")
        for dependency in graph[node]:
            visit(dependency, path | {node})
    for node in graph:
        visit(node, set())
