"""Rating-based world difficulty and monthly sponsorship, without mutating base stats."""

from dataclasses import dataclass, replace
import math


class WorldLevelError(ValueError):
    pass


@dataclass(frozen=True)
class WorldLevel:
    level: int
    required_rating: float
    enemy_multiplier: float
    sponsor_funds: int


def validate_world_level(row):
    if not isinstance(row, WorldLevel) or type(row.level) is not int or row.level < 1:
        raise WorldLevelError("世界レベルは1以上の整数にしてください。")
    if type(row.required_rating) not in (int, float) or not math.isfinite(row.required_rating) or row.required_rating < 0:
        raise WorldLevelError("必要レートは0以上の有限の数値にしてください。")
    if type(row.enemy_multiplier) not in (int, float) or not math.isfinite(row.enemy_multiplier) or row.enemy_multiplier <= 0:
        raise WorldLevelError("敵倍率は0より大きい有限の数値にしてください。")
    if type(row.sponsor_funds) is not int or row.sponsor_funds < 0:
        raise WorldLevelError("スポンサー資金は0以上の円単位の整数にしてください。")


def configured_world_levels():
    from realtime_season_world_levels import WORLD_LEVELS

    if not isinstance(WORLD_LEVELS, (list, tuple)) or not WORLD_LEVELS:
        raise WorldLevelError("WORLD_LEVELS に世界レベル設定を1件以上指定してください。")
    result = []
    for row in WORLD_LEVELS:
        if not isinstance(row, dict) or set(row) != {"レベル", "必要レート", "敵倍率", "スポンサー資金"}:
            raise WorldLevelError("世界レベルには「レベル」「必要レート」「敵倍率」「スポンサー資金」を設定してください。")
        row = WorldLevel(*(row[k] for k in ("レベル", "必要レート", "敵倍率", "スポンサー資金")))
        validate_world_level(row)
        result.append(row)
    result.sort(key=lambda row: row.level)
    if result[0].required_rating != 0:
        raise WorldLevelError("最低世界レベルの必要レートは0にしてください。")
    for previous, current in zip(result, result[1:]):
        if current.level == previous.level or current.required_rating <= previous.required_rating:
            raise WorldLevelError("レベルは重複させず、必要レートは高いレベルほど大きくしてください。")
        if current.enemy_multiplier < previous.enemy_multiplier or current.sponsor_funds < previous.sponsor_funds:
            raise WorldLevelError("敵倍率とスポンサー資金は高いレベルほど同じか大きい値にしてください。")
    return tuple(result)


def world_level_for_rating(rating):
    if type(rating) not in (int, float) or not math.isfinite(rating) or rating < 0:
        raise WorldLevelError("レートは0以上の有限の数値にしてください。")
    return max((row for row in configured_world_levels() if rating >= row.required_rating),
               key=lambda row: row.level)


def world_level_from_save(row, *, legacy=False):
    """Preserve an old tournament's locked effects until its completion."""
    if not isinstance(row, dict):
        raise WorldLevelError("固定された世界レベルの形式が不正です。")
    row = dict(row)
    if legacy and "top_percent" in row:
        percent = row.pop("top_percent")
        if type(percent) not in (int, float) or not math.isfinite(percent) or not 0 < percent <= 100:
            raise WorldLevelError("旧セーブの世界レベルの上位%が不正です。")
        threshold = next((level.required_rating for level in configured_world_levels()
                          if level.level == row.get("level")), 0)
        row.setdefault("required_rating", threshold)
    result = WorldLevel(**row)
    validate_world_level(result)
    return result


def scale_enemy_player(player, multiplier):
    if multiplier == 1:
        return player
    changes = {name: getattr(player, name) * multiplier
               for name in ("hs_pct", "dodge_pct", "hit_pct", "iq", "reaction", "influence", "mental")}
    for name in ("hs_pct", "dodge_pct"):
        changes[name] = min(1.0, changes[name])
    changes["mental"] = min(10.0, changes["mental"])
    return replace(player, **changes)
