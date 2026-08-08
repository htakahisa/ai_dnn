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

    @property
    def condition_variance(self) -> float:
        """「調子の波」の意味を明示した互換用エイリアス。"""
        return self.form_variance


CHARACTER_TABLE: Dict[str, CharacterStats] = {
    "Tortlilyan": CharacterStats("Tortlilyan", 0.23, 0.39, 123, 0.9, 156, "タイガー", 50, 10),
    "いぐるん": CharacterStats("いぐるん",0.33,0.22,75,0.77,131,"シーカー",55,1),
    "ろびぃな": CharacterStats("ろびぃな",0.25,0.55,80,0.65,122,"スモーカー",50,8),
    "夢の街": CharacterStats("夢の街",0.35,0.23,80,0.76,115,"フラッシュ",50,4),
    "えんぺん": CharacterStats("えんぺん",0.5,0.17,85,0.75,125,"フラッシュ",65,7),
}

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
