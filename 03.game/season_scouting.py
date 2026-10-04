"""Calendar-based scouting allowances, independent of monthly settlement."""

from datetime import date

from season_competitions import parse_date
import realtime_season_scouting as settings


def season_window(day, periods):
    md = day.strftime("%m-%d")
    for first, last in periods:
        if (first <= md <= last if first <= last else md >= first or md <= last):
            year = day.year - int(first > last and md <= last)
            # February 29 settings also work in ordinary years.
            def boundary(year, value):
                month, number = map(int, value.split("-"))
                if month == 2 and number == 29:
                    from calendar import monthrange
                    number = monthrange(year, month)[1]
                return date(year, month, number)
            return boundary(year, first), boundary(year + int(first > last), last)
    return None


def remaining(state, team_id):
    window = season_window(state.date, state.in_season_periods)
    if window is None:
        return None
    used = sum(1 for day, club in state.scout_uses
               if club == team_id and window[0] <= parse_date(day) <= window[1])
    return max(0, settings.IN_SEASON_SCOUT_LIMIT - used)


def validate_uses(uses, current):
    if type(settings.IN_SEASON_SCOUT_LIMIT) is not int or settings.IN_SEASON_SCOUT_LIMIT < 0:
        raise ValueError("インシーズンのスカウト上限は0以上の整数にしてください。")
    if not isinstance(uses, tuple):
        raise ValueError("スカウト実績の形式が不正です。")
    for row in uses:
        if (not isinstance(row, tuple) or len(row) != 2
                or not isinstance(row[1], str) or not row[1]
                or parse_date(row[0]) > current):
            raise ValueError("スカウト実績の日付・チームIDが不正です。")
