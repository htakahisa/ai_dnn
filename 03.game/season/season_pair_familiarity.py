"""Cumulative co-membership, independent of combat combos and game RNG."""

from dataclasses import dataclass, replace
from itertools import combinations
import math
from random import Random
from uuid import NAMESPACE_URL, uuid5
import json

import realtime_season_pair_familiarity as settings


def pair_key(first, second):
    if first == second:
        raise ValueError("同じ選手のペアは作れません。")
    return tuple(sorted((first, second)))


def validate_settings():
    if type(settings.pair_familiarity_enabled) is not bool:
        raise ValueError("pair_familiarity_enabled は True / False にしてください。")
    for value in (settings.TAU_DAYS, settings.M0, settings.M_MAX):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("練度の日数と倍率は正の有限の数値にしてください。")
    if settings.M_MAX < settings.M0:
        raise ValueError("M_MAX は M0 以上にしてください。")
    if (len(settings.STAGES) != 5 or settings.STAGES[0][1] != 0
            or any(not isinstance(name, str) or not name or type(v) not in (int, float)
                   or not math.isfinite(v) or not 0 <= v < 1 for name, v in settings.STAGES)
            or any(a[1] >= b[1] for a, b in zip(settings.STAGES, settings.STAGES[1:]))):
        raise ValueError("練度の5段階は0から始まる昇順のしきい値で設定してください。")
    if (type(settings.INITIAL_SEED) is not int
            or type(settings.AI_INITIAL_MIN_DAYS) is not int
            or type(settings.AI_INITIAL_MAX_DAYS) is not int
            or not 0 < settings.AI_INITIAL_MIN_DAYS <= settings.AI_INITIAL_MAX_DAYS
            or type(settings.AI_PAIR_VARIATION) not in (int, float)
            or not math.isfinite(settings.AI_PAIR_VARIATION) or not 0 <= settings.AI_PAIR_VARIATION < 1):
        raise ValueError("AIの初期日数・シード・ばらつきが不正です。")
    for value in (settings.MAX_NEWS_PER_DAY, settings.PARTNER_LIMIT, settings.RANKING_LIMIT):
        if type(value) is not int or value < 0:
            raise ValueError("練度の表示・通知件数は0以上の整数にしてください。")
    if (not isinstance(settings.STAGE_COLORS, (list, tuple)) or len(settings.STAGE_COLORS) != 5
            or any(not isinstance(color, str) or not color for color in settings.STAGE_COLORS)):
        raise ValueError("練度の段階色を5色設定してください。")


def familiarity(days):
    return -math.expm1(-days / settings.TAU_DAYS)


def stage_index(value):
    return max(i for i, (_, threshold) in enumerate(settings.STAGES) if value >= threshold)


def stage(days):
    return settings.STAGES[stage_index(familiarity(days))][0]


def team_metrics(pair_days, starters):
    names = tuple(starters)
    if len(names) != 5 or len(set(names)) != 5:
        return None, 1.0
    c = sum(familiarity(pair_days.get(pair_key(a, b), 0)) for a, b in combinations(names, 2)) / 10
    return c, settings.M0 + (settings.M_MAX - settings.M0) * c


def memberships(state):
    result = {p.name: state.club_id for p in state.owned_players}
    result.update((p.name, club.id) for club in state.opponent_teams for p in club.players)
    return result


def initial_pair_days(state):
    days = dict(state.pair_days)
    for club in state.opponent_teams:
        rng = Random(f"pair-familiarity:{state.pair_familiarity_seed}:{club.id}")
        baseline = rng.randint(settings.AI_INITIAL_MIN_DAYS, settings.AI_INITIAL_MAX_DAYS)
        regular = set(club.regular_members or club.members)
        retained = sorted(regular.intersection(club.members))
        for key in combinations(retained, 2):
            days.setdefault(key, max(1, round(baseline * rng.uniform(
                1 - settings.AI_PAIR_VARIATION, 1 + settings.AI_PAIR_VARIATION))))
    return replace(state, pair_days=days)


def advance_pair_days(state):
    days = dict(state.pair_days)
    groups = (tuple(p.name for p in state.owned_players), *(c.members for c in state.opponent_teams))
    for names in groups:
        for first, second in combinations(names, 2):
            key = pair_key(first, second)
            days[key] = days.get(key, 0) + 1
    return replace(state, pair_days=days)


def validate_pair_days(pair_days):
    if not isinstance(pair_days, dict):
        raise ValueError("pair_days は疎な辞書で保存してください。")
    for key, days in pair_days.items():
        if (not isinstance(key, tuple) or len(key) != 2
                or any(not isinstance(n, str) or not n for n in key)
                or key[0] >= key[1] or type(days) is not int or days <= 0):
            raise ValueError("ペア練度の選手キーまたは日数が不正です。")


def encode_pair_days(pair_days):
    return [[a, b, days] for (a, b), days in sorted(pair_days.items())]


def decode_pair_days(rows):
    if not isinstance(rows, list):
        raise ValueError("ペア練度の保存形式が不正です。")
    result = {}
    for row in rows:
        if (not isinstance(row, list) or len(row) != 3 or any(not isinstance(n, str) for n in row[:2])
                or row[0] >= row[1] or (row[0], row[1]) in result
                or type(row[2]) is not int or row[2] <= 0):
            raise ValueError("ペア練度の保存行が重複または不正です。")
        result[(row[0], row[1])] = row[2]
    return result


def history_metrics(state, run=None):
    registered = {t.id: tuple(p.name for p in t.players) for t in run.entrants} if run else {}
    result = []
    for team_id, name in ((state.club_id, state.team_name), *((c.id, c.name) for c in state.opponent_teams)):
        starters = registered.get(team_id, state.pair_starters(team_id))
        c, m = team_metrics(state.pair_days, starters)
        result.append({"チームID": team_id, "チーム名": name, "スタメン": list(starters),
                       "C": c, "M": m, "適用倍率": m if settings.pair_familiarity_enabled else 1.0})
    return result


@dataclass(frozen=True)
class PairNews:
    id: str
    date: str
    kind: str
    pair: tuple[str, str]
    stage: str
    days: int
    message: str
    player_priority: bool


def _news(state, key, kind, days, player_priority):
    label = stage(days)
    identity = json.dumps((state.date.isoformat(), kind, key, label), ensure_ascii=False)
    identifier = "pair-" + uuid5(NAMESPACE_URL, identity).hex
    message = (f"{key[0]}と{key[1]}が「{label}」になりました（通算{days:,}日）。" if kind == "pair_upgrade"
               else f"「{label}」の{key[0]}と{key[1]}が移籍・退団で別々になりました（通算{days:,}日）。")
    return PairNews(identifier, state.date.isoformat(), kind, key, label, days, message, player_priority)


def _append_news(state, events):
    current = {e.id: e for e in state.pair_news if e.date == state.date.isoformat()}
    current.update((e.id, e) for e in events)
    chosen = sorted(current.values(), key=lambda e: (not e.player_priority, -e.days, e.id))[:settings.MAX_NEWS_PER_DAY]
    older = tuple(e for e in state.pair_news if e.date != state.date.isoformat())
    return replace(state, pair_news=(*older, *chosen))


def add_separation_news(state, before):
    if not settings.pair_familiarity_enabled or not before:
        return state
    after = memberships(state)
    events = []
    for key, days in state.pair_days.items():
        a, b = key
        if (before.get(a) is not None and before.get(a) == before.get(b)
                and not (after.get(a) is not None and after.get(a) == after.get(b))
                and stage_index(familiarity(days)) >= 2):
            priority = state.club_id in (before.get(a), before.get(b), after.get(a), after.get(b))
            events.append(_news(state, key, "pair_separation", days, priority))
    return _append_news(state, events) if events else state


def daily_news(state, previous):
    if not settings.pair_familiarity_enabled:
        return state
    state = add_separation_news(state, memberships(previous))
    owners = memberships(state)
    events = []
    for key, days in state.pair_days.items():
        old_index = stage_index(familiarity(previous.pair_days.get(key, 0)))
        new_index = stage_index(familiarity(days))
        a, b = key
        together = owners.get(a) is not None and owners.get(a) == owners.get(b)
        player = owners.get(a) == state.club_id and owners.get(b) == state.club_id
        if together and new_index > old_index and new_index >= 2 and (player or new_index == 4):
            events.append(_news(state, key, "pair_upgrade", days, player))
    return _append_news(state, events) if events else state


def validate_news(news, current_date):
    if not isinstance(news, tuple):
        raise ValueError("ペアニュースの形式が不正です。")
    from datetime import date
    ids = set()
    last_date = None
    for item in news:
        if (not isinstance(item, PairNews) or not isinstance(item.id, str) or not item.id or item.id in ids
                or item.kind not in ("pair_upgrade", "pair_separation")
                or not isinstance(item.pair, tuple) or len(item.pair) != 2
                or any(not isinstance(n, str) or not n for n in item.pair) or item.pair[0] >= item.pair[1]
                or type(item.days) is not int or item.days <= 0
                or not isinstance(item.stage, str) or not item.stage
                or not isinstance(item.message, str) or not item.message
                or type(item.player_priority) is not bool):
            raise ValueError("ペアニュースの項目が不正です。")
        day = date.fromisoformat(item.date)
        if day > current_date or last_date is not None and day < last_date:
            raise ValueError("ペアニュースの日付が不正です。")
        last_date = day
        ids.add(item.id)
