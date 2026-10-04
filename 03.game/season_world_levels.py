"""Rank-based world difficulty and monthly sponsorship, without mutating base stats."""

from dataclasses import dataclass, replace
import math


class WorldLevelError(ValueError):
    pass


@dataclass(frozen=True)
class WorldLevel:
    level: int
    top_percent: float
    enemy_multiplier: float
    sponsor_funds: int


def validate_world_level(row):
    if not isinstance(row, WorldLevel) or type(row.level) is not int or row.level < 1:
        raise WorldLevelError("世界レベルは1以上の整数にしてください。")
    if type(row.top_percent) not in (int, float) or not math.isfinite(row.top_percent) or not 0 < row.top_percent <= 100:
        raise WorldLevelError("上位%は0より大きく100以下の有限の数値にしてください。")
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
        if not isinstance(row, dict) or set(row) != {"レベル", "上位%", "敵倍率", "スポンサー資金"}:
            raise WorldLevelError("世界レベルには「レベル」「上位%」「敵倍率」「スポンサー資金」を設定してください。")
        level, percent, multiplier, funds = (row[k] for k in ("レベル", "上位%", "敵倍率", "スポンサー資金"))
        row = WorldLevel(level, percent, multiplier, funds)
        validate_world_level(row)
        result.append(row)
    result.sort(key=lambda row: row.level)
    if result[0].top_percent != 100:
        raise WorldLevelError("最低世界レベルの上位%は100にしてください。")
    for previous, current in zip(result, result[1:]):
        if current.level == previous.level or current.top_percent >= previous.top_percent:
            raise WorldLevelError("レベルは重複させず、上位%は高いレベルほど小さくしてください。")
        if current.enemy_multiplier < previous.enemy_multiplier or current.sponsor_funds < previous.sponsor_funds:
            raise WorldLevelError("敵倍率とスポンサー資金は高いレベルほど同じか大きい値にしてください。")
    return tuple(result)


def world_level_for_rank(rank, team_count):
    from decimal import Decimal

    # Compare before division so exact percentile boundaries remain inclusive.
    return max((row for row in configured_world_levels()
                if Decimal(rank * 100) <= Decimal(str(row.top_percent)) * team_count), key=lambda row: row.level)


def scale_enemy_player(player, multiplier):
    if multiplier == 1:
        return player
    changes = {name: getattr(player, name) * multiplier
               for name in ("hs_pct", "dodge_pct", "hit_pct", "iq", "reaction", "influence", "mental")}
    for name in ("hs_pct", "dodge_pct"):
        changes[name] = min(1.0, changes[name])
    changes["mental"] = min(10.0, changes["mental"])
    return replace(player, **changes)
