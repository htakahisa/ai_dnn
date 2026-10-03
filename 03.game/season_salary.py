"""Competition K/D salary rules, file-change-aware caches and saved snapshots."""

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from enum import Enum
from functools import lru_cache
import json
import logging
import math
from pathlib import Path


class SalaryMode(str, Enum):
    STATIC = "STATIC"
    KD_DYNAMIC = "KD_DYNAMIC"


SALARY_MODE_LABELS = {"静的月給（従来）": SalaryMode.STATIC, "成績連動月給（K/D）": SalaryMode.KD_DYNAMIC}


class SalaryDataError(ValueError):
    pass


@dataclass(frozen=True)
class SalarySettings:
    results_directory: str
    a: float = 30
    min_games: int = 10
    max_change: float = .20
    fixed_salaries: tuple[tuple[str, int], ...] = ()
    excluded_files: tuple[str, ...] = ()

    def validate(self):
        if not isinstance(self.results_directory, str) or not self.results_directory.strip():
            raise SalaryDataError("成績の保存ディレクトリが不正です。")
        if type(self.a) not in (int, float) or not math.isfinite(self.a) or self.a <= 0:
            raise SalaryDataError("給与計算のAは0より大きい有限の数値にしてください。")
        if type(self.min_games) is not int or self.min_games < 1:
            raise SalaryDataError("MIN_GAMESは1以上の整数にしてください。")
        if type(self.max_change) not in (int, float) or not math.isfinite(self.max_change) or not 0 <= self.max_change <= 1:
            raise SalaryDataError("MAX_MONTHLY_CHANGEは0～1にしてください。")
        names = set()
        if not isinstance(self.fixed_salaries, tuple):
            raise SalaryDataError("固定月給テーブルの形式が不正です。")
        for row in self.fixed_salaries:
            if not isinstance(row, tuple) or len(row) != 2:
                raise SalaryDataError("固定月給テーブルの形式が不正です。")
            name, salary = row
            if not isinstance(name, str) or not name.strip() or name in names or type(salary) is not int or salary < 0:
                raise SalaryDataError("固定月給は重複のない選手名と0以上の円単位の整数で設定してください。")
            names.add(name)
        if not isinstance(self.excluded_files, tuple) or any(
                not isinstance(name, str) or not name.endswith(".json") or Path(name).name != name
                for name in self.excluded_files):
            raise SalaryDataError("除外する成績はJSONファイル名のタプルで設定してください。")


@dataclass(frozen=True)
class CompetitionTotals:
    name: str
    kills: int = 0
    deaths: int = 0
    games: int = 0


@dataclass(frozen=True)
class SalaryRecord:
    name: str
    static_salary: int
    monthly_salary: int
    kills: int = 0
    deaths: int = 0
    games: int = 0
    kd: float = 1.0
    percentile: float | None = None
    fixed: bool = False

    def validate(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise SalaryDataError("月給記録の選手名が不正です。")
        for value in (self.static_salary, self.monthly_salary, self.kills, self.deaths, self.games):
            if type(value) is not int or value < 0:
                raise SalaryDataError("月給・成績の記録は0以上の整数にしてください。")
        if type(self.kd) not in (int, float) or not math.isfinite(self.kd) or self.kd < 0:
            raise SalaryDataError("月給記録のK/Dが不正です。")
        if self.percentile is not None and (type(self.percentile) not in (int, float)
                or not math.isfinite(self.percentile) or not 0 <= self.percentile <= 1):
            raise SalaryDataError("月給記録のパーセンタイルが不正です。")
        if type(self.fixed) is not bool:
            raise SalaryDataError("固定月給の記録が不正です。")


def configured_salary_settings():
    import season_salary_config as config
    if not isinstance(config.FIXED_MONTHLY_SALARIES, dict):
        raise SalaryDataError("FIXED_MONTHLY_SALARIESは選手名と月給の辞書にしてください。")
    settings = SalarySettings(str(Path(config.COMPETITION_RESULTS_DIR).resolve()), config.A,
                              config.MIN_GAMES, config.MAX_MONTHLY_CHANGE,
                              tuple(config.FIXED_MONTHLY_SALARIES.items()), config.EXCLUDED_RESULT_FILES)
    settings.validate()
    return settings


def smoothed_kd(kills, deaths, a=30):
    return (kills + a) / (deaths + a)


def salary_from_percentile(p, lo=90, mid=150, hi=300):
    """Return the unrounded salary in units of 10,000 yen."""
    p = min(max(p, 0.0), 1.0)
    if p <= .5:
        return lo * (mid / lo) ** (p / .5)
    return mid * (hi / mid) ** ((p - .5) / .5)


def rounded_salary(p):
    return int(Decimal(str(salary_from_percentile(p))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) * 10_000


def clamp_change(new, prev, c=.2):
    if prev is None:
        return new
    # Clamp in yen after the 10,000-yen rounding. Directed rounding keeps
    # the final integer strictly inside the permitted percentage interval.
    lower = int((Decimal(prev) * (1 - Decimal(str(c)))).to_integral_value(rounding=ROUND_CEILING))
    upper = int((Decimal(prev) * (1 + Decimal(str(c)))).to_integral_value(rounding=ROUND_FLOOR))
    return min(max(new, lower), upper)


def _percentiles(totals, settings):
    eligible = sorted((smoothed_kd(t.kills, t.deaths, settings.a), t.name)
                      for t in totals if t.games >= settings.min_games)
    result = {}
    index, count = 0, len(eligible)
    while index < count:
        end = index + 1
        while end < count and eligible[end][0] == eligible[index][0]:
            end += 1
        p = ((index + end - 1) / 2) / (count - 1) if count >= 2 else .5
        for _, name in eligible[index:end]:
            result[name] = p
        index = end
    return result


def calculate_salaries(static_salaries, totals, settings, previous=None):
    settings.validate()
    totals = tuple(totals)
    stats = {t.name: t for t in totals}
    percentiles = _percentiles(totals, settings)
    fixed = dict(settings.fixed_salaries)
    records = []
    for name, static_salary in sorted(static_salaries.items()):
        stat = stats.get(name, CompetitionTotals(name))
        p = percentiles.get(name)
        if name in fixed:
            salary = fixed[name]
        elif p is None:
            # Ineligible players keep their static salary, including snapshots
            # saved before the catalog was edited.
            salary = static_salary
        else:
            salary = rounded_salary(p)
        if name not in fixed and previous is not None:
            salary = clamp_change(salary, previous.get(name), settings.max_change)
        record = SalaryRecord(name, static_salary, salary, stat.kills, stat.deaths, stat.games,
                              smoothed_kd(stat.kills, stat.deaths, settings.a), p, name in fixed)
        record.validate()
        records.append(record)
    return tuple(records)


def _file_signature(directory, excluded_files=()):
    path = Path(directory)
    if not path.exists():
        return ()
    if not path.is_dir():
        raise SalaryDataError("成績の保存先はディレクトリにしてください。")
    signature = []
    for p in sorted(path.glob("*.json")):
        if p.name == "team_ratings.json" or p.name in excluded_files:
            continue
        stat = p.stat()
        signature.append((p.name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _rows_from_result(data):
    if "evaluation" in data:
        return []
    board = data.get("player_leaderboards")
    if isinstance(board, dict) and "all_players" in board:
        if not isinstance(board["all_players"], list):
            raise SalaryDataError("all_playersの形式が不正です。")
        return board["all_players"]
    # Older results without all_players still contain complete per-map stats.
    maps = data.get("maps", [])
    for key in ("matches", "bracket_matches"):
        for series in data.get(key, []):
            maps = [*maps, *series.get("maps", [])]
    rows = []
    for game_map in maps:
        for stat in game_map.get("player_stats", []):
            rows.append({"player": stat["name"], "kills": stat["kills"], "deaths": stat["deaths"], "maps": 1})
    return rows


def _read_result(path):
    # Official results put the authoritative leaderboard after enormous map
    # logs. Read that top-level field and the remaining JSON without building
    # millions of unrelated log objects. Compact/legacy JSON uses the fallback.
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - 262_144))
        tail = stream.read()
        marker = b'\n  "player_leaderboards":'
        position = tail.rfind(marker)
        if size > len(tail) and position >= 0:
            return json.loads(b'{"player_leaderboards":' + tail[position + len(marker):])
        stream.seek(0)
        return json.load(stream)


@lru_cache(maxsize=8)
def _competition_totals(directory, signature):
    totals = defaultdict(lambda: [0, 0, 0])
    for filename, _, _ in signature:
        path = Path(directory) / filename
        try:
            data = _read_result(path)
            if not isinstance(data, dict):
                raise SalaryDataError("結果ファイルはJSONオブジェクトにしてください。")
            for row in _rows_from_result(data):
                name = row.get("player")
                values = [row.get("kills"), row.get("deaths"), row.get("maps")]
                if not isinstance(name, str) or not name.strip() or any(type(v) is not int or v < 0 for v in values):
                    raise SalaryDataError("選手のkills / deaths / mapsの形式が不正です。")
                totals[name] = [a + b for a, b in zip(totals[name], values)]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise SalaryDataError(f"コンペティション成績を読み込めません: {path.name}: {exc}") from exc
    return tuple(CompetitionTotals(name, *values) for name, values in sorted(totals.items()))


@lru_cache(maxsize=8)
def _salary_targets(static_items, settings, signature):
    return calculate_salaries(dict(static_items), _competition_totals(settings.results_directory, signature), settings)


def salary_records(static_salaries, settings, previous=None):
    """Recheck the file signature; cache work only while results are unchanged."""
    try:
        signature = _file_signature(settings.results_directory, settings.excluded_files)
        targets = _salary_targets(tuple(sorted(static_salaries.items())), settings, signature)
    except OSError as exc:
        raise SalaryDataError(f"コンペティション成績を確認できません: {exc}") from exc
    if previous is None:
        return targets
    from dataclasses import replace
    return tuple(replace(r, monthly_salary=clamp_change(r.monthly_salary, previous.get(r.name), settings.max_change))
                 if not r.fixed else r for r in targets)


def invalidate_salary_cache():
    _competition_totals.cache_clear()
    _salary_targets.cache_clear()


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Salary] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.propagate = False


def log_salary_change(name, old, new, record, *, contract=False):
    if old != new:
        LOGGER.info("%s %s: %s -> %s yen / kd=%.6f / p=%s", name,
                    "契約月給" if contract else "基本月給", old, new, record.kd,
                    "対象外" if record.percentile is None else f"{record.percentile:.6f}")
