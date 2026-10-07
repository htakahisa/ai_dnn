"""Per-player paid IQ research and aim practice."""

from dataclasses import dataclass
from decimal import Decimal
import math


MAX_TRAINING_LEVEL = 30
TRAINING_FIELDS = {"research": ("研究", "research_level", "iq"),
                   "aim_lab": ("エイムラボ", "aim_lab_level", "hit_pct")}


@dataclass(frozen=True)
class TrainingTerms:
    kind: str
    title: str
    level_field: str
    stat_field: str
    next_level: int
    cost: int
    before: float
    after: float


def training_terms(player, kind):
    from realtime_season import SeasonSaveError
    from realtime_season_training import TRAINING

    if not isinstance(kind, str) or kind not in TRAINING_FIELDS:
        raise SeasonSaveError("育成施設を選択してください。")
    title, level_field, stat_field = TRAINING_FIELDS[kind]
    row = TRAINING.get(title) if isinstance(TRAINING, dict) else None
    if not isinstance(row, dict):
        raise SeasonSaveError(f"{title}の育成設定が見つかりません。")
    growth, costs = row.get("上昇量"), row.get("費用")
    if (type(growth) not in (int, float) or not math.isfinite(growth) or growth <= 0
            or not isinstance(costs, (list, tuple)) or len(costs) != MAX_TRAINING_LEVEL
            or any(type(cost) is not int or cost <= 0 for cost in costs)
            or any(right <= left for left, right in zip(costs, costs[1:]))):
        raise SeasonSaveError(f"{title}は正の上昇量と、順に増える{MAX_TRAINING_LEVEL}レベル分の費用（円・整数）を設定してください。")
    level = getattr(player, level_field)
    if level >= MAX_TRAINING_LEVEL:
        raise SeasonSaveError(f"{title}は最大レベル{MAX_TRAINING_LEVEL}です。")
    before = getattr(player, stat_field)
    amount = Decimal(str(growth)) / (100 if kind == "aim_lab" else 1)
    after = float(Decimal(str(before)) + amount)
    if not math.isfinite(after) or after <= before:
        raise SeasonSaveError("育成後の能力値を増加させられません。上昇量を確認してください。")
    return TrainingTerms(kind, title, level_field, stat_field, level + 1, costs[level], before, after)
