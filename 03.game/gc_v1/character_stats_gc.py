from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CharacterStats:
    name: str
    hs_pct: float
    dodge_pct: float
    iq: float
    hit_pct: float
    reaction: float
    role: str
    influence: float
    # スプレッドシートの「調子の波」。0が最も安定、値が大きいほど波が大きい。
    form_variance: float = 0.0
    # 0～10。高いほど逆境・長期戦・シリーズ劣勢の影響を受けにくい。
    mental: float = 5.0

    @property
    def condition_variance(self) -> float:
        """「調子の波」の意味を明示した互換用エイリアス。"""
        return self.form_variance


CHARACTER_TABLE: Dict[str, CharacterStats] = {
    "Xdll": CharacterStats("Xdll", 0.23, 0.24, 125, 0.78, 140, "シーカー", 45, 7, 4),
    "SyouTa": CharacterStats("SyouTa", 0.45, 0.2, 90, 0.75, 140, "タイガー", 95, 8, 9),
    "Absol": CharacterStats("Absol", 0.15, 0.31, 130, 0.87, 155, "タイガー", 50, 10, 3),
    "eKo": CharacterStats("eKo", 0.3, 0.21, 85, 0.76, 110, "タイガー", 60, 5, 8),
    "SugarZ3ro": CharacterStats(
        "SugarZ3ro", 0.29, 0.18, 120, 0.67, 93, "スモーカー", 40, 5, 5
    ),
}

GC_ROSTER_ORDER = ["Xdll", "Syouta", "Absol", "eKo", "SugarZ3ro"]

# 旧コードとの互換用エイリアス
CHARACTER_STATS = CHARACTER_TABLE
character_stats = CHARACTER_TABLE


def get_by_name(name: str) -> Optional[CharacterStats]:
    return CHARACTER_TABLE.get(name)


def get_stats(name: str) -> Optional[CharacterStats]:
    return get_by_name(name)


def all_characters() -> List[CharacterStats]:
    return list(CHARACTER_TABLE.values())


def all_names() -> List[str]:
    return list(CHARACTER_TABLE.keys())
