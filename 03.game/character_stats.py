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
    "Chronicle": CharacterStats(
        "Chronicle", 0.35, 0.23, 105, 0.78, 135, "フラッシュ", 80, 3
    ),
    "Demon1": CharacterStats("Demon1", 0.5, 0.13, 78, 0.85, 145, "スモーカー", 150, 4),
    "something": CharacterStats(
        "something", 0.35, 0.26, 65, 0.88, 165, "タイガー", 85, 7
    ),
    "Leo": CharacterStats("Leo", 0.27, 0.22, 150, 0.83, 135, "シーカー", 65, 1),
    "Alfajer": CharacterStats("Alfajer", 0.45, 0.13, 60, 0.91, 145, "タイガー", 60, 4),
    "Derke": CharacterStats("Derke", 0.35, 0.24, 90, 0.75, 115, "タイガー", 90, 8),
    "Boaster": CharacterStats("Boaster", 0.2, 0.12, 175, 0.62, 96, "スモーカー", 50, 5),
    "Aspas": CharacterStats("Aspas", 0.4, 0.24, 90, 0.87, 120, "タイガー", 130, 10),
    "F0rsakeN": CharacterStats(
        "F0rsakeN", 0.27, 0.14, 125, 0.69, 95, "スモーカー", 50, 3
    ),
    "Jinggg": CharacterStats("Jinggg", 0.3, 0.26, 68, 0.71, 90, "タイガー", 50, 4),
    "d4v41": CharacterStats("d4v41", 0.35, 0.17, 70, 0.71, 105, "シーカー", 40, 5),
    "Sato": CharacterStats("Sato", 0.4, 0.19, 67, 0.73, 105, "フラッシュ", 60, 6),
    "jawgemo": CharacterStats("jawgemo", 0.28, 0.31, 93, 0.67, 135, "タイガー", 75, 8),
    "valyn": CharacterStats("valyn", 0.26, 0.13, 140, 0.64, 95, "スモーカー", 40, 5),
    "Ethan": CharacterStats("Ethan", 0.3, 0.18, 150, 0.8, 90, "シーカー", 55, 4),
    "Wo0t": CharacterStats("Wo0t", 0.4, 0.16, 78, 0.75, 120, "タイガー", 75, 9),
    "kaajak": CharacterStats("kaajak", 0.35, 0.2, 115, 0.76, 100, "フラッシュ", 70, 7),
    "Meiy": CharacterStats("Meiy", 0.35, 0.22, 70, 0.73, 110, "タイガー", 90, 7),
    "Verno": CharacterStats("Verno", 0.35, 0.16, 86, 0.7, 95, "シーカー", 80, 7),
    "Sayonara": CharacterStats("Sayonara", 0.33, 0.19, 95, 0.7, 90, "シーカー", 80, 5),
    "Jamppi": CharacterStats("Jamppi", 0.27, 0.15, 125, 0.68, 85, "フラッシュ", 55, 3),
    "Boostio": CharacterStats(
        "Boostio", 0.28, 0.15, 128, 0.68, 85, "スモーカー", 70, 6
    ),
    "Primmie": CharacterStats("Primmie", 0.47, 0.24, 59, 0.79, 98, "タイガー", 90, 5),
    "Meteor": CharacterStats("Meteor", 0.37, 0.29, 73, 0.78, 95, "フラッシュ", 75, 8),
    "Laz": CharacterStats("Laz", 0.4, 0.14, 70, 0.75, 86, "フラッシュ", 80, 6),
    "ZMJKK": CharacterStats("ZMJKK", 0.3, 0.17, 65, 0.71, 150, "タイガー", 90, 9),
    "Brawk": CharacterStats("Brawk", 0.32, 0.15, 90, 0.69, 95, "シーカー", 90, 6),
    "HYUNMIN": CharacterStats(
        "HYUNMIN", 0.35, 0.24, 76, 0.75, 110, "フラッシュ", 70, 5
    ),
    "Flashback": CharacterStats(
        "Flashback", 0.38, 0.22, 79, 0.77, 130, "フラッシュ", 70, 5
    ),
    "Mako": CharacterStats("Mako", 0.28, 0.16, 120, 0.7, 85, "スモーカー", 50, 5),
    "t3xture": CharacterStats("t3xture", 0.33, 0.19, 93, 0.75, 150, "タイガー", 75, 7),
    "trent": CharacterStats("trent", 0.3, 0.18, 120, 0.71, 85, "シーカー", 40, 5),
    "leaf": CharacterStats("leaf", 0.38, 0.17, 89, 0.74, 80, "フラッシュ", 40, 5),
    "keiko": CharacterStats("keiko", 0.31, 0.19, 89, 0.72, 90, "スモーカー", 70, 8),
    "Rb": CharacterStats("Rb", 0.29, 0.23, 110, 0.69, 80, "スモーカー", 40, 4),
    "stax": CharacterStats("stax", 0.29, 0.15, 130, 0.69, 80, "フラッシュ", 70, 4),
    "tex": CharacterStats("tex", 0.35, 0.19, 85, 0.74, 95, "フラッシュ", 60, 7),
    "Mazino": CharacterStats("Mazino", 0.33, 0.18, 100, 0.72, 94, "スモーカー", 60, 7),
    "Zekken": CharacterStats("Zekken", 0.35, 0.25, 85, 0.7, 105, "フラッシュ", 100, 8),
    "BABYBAY": CharacterStats("BABYBAY", 0.4, 0.15, 85, 0.73, 100, "タイガー", 55, 7),
    "Dep": CharacterStats("Dep", 0.33, 0.22, 75, 0.73, 100, "タイガー", 50, 5),
    "SugarZ3ro": CharacterStats(
        "SugarZ3ro", 0.29, 0.18, 120, 0.65, 93, "スモーカー", 40, 5
    ),
    "Buzz": CharacterStats("Buzz", 0.3, 0.23, 80, 0.72, 100, "タイガー", 60, 5),
    "TenZ": CharacterStats("TenZ", 0.38, 0.23, 105, 0.76, 120, "スモーカー", 170, 9),
    "eggsterr": CharacterStats(
        "eggsterr", 0.3, 0.22, 88, 0.7, 95, "フラッシュ", 50, 10
    ),
    "Rossy": CharacterStats("Rossy", 0.31, 0.18, 89, 0.7, 100, "シーカー", 50, 9),
    "Rarga": CharacterStats("Rarga", 0.35, 0.19, 75, 0.75, 120, "タイガー", 50, 7),
    "Lysoar": CharacterStats("Lysoar", 0.33, 0.17, 105, 0.71, 100, "スモーカー", 50, 7),
    "Smoggy": CharacterStats("Smoggy", 0.38, 0.19, 75, 0.73, 120, "スモーカー", 50, 8),
    "CHICHOO": CharacterStats(
        "CHICHOO", 0.32, 0.18, 80, 0.74, 100, "スモーカー", 50, 9
    ),
    "crashies": CharacterStats(
        "crashies", 0.28, 0.17, 140, 0.68, 80, "シーカー", 45, 0
    ),
    "FNS": CharacterStats("FNS", 0.17, 0.13, 190, 0.63, 70, "スモーカー", 10, 0),
    "nAts": CharacterStats("nAts", 0.29, 0.18, 130, 0.72, 88, "スモーカー", 50, 9),
    "Lar0k": CharacterStats("Lar0k", 0.34, 0.18, 85, 0.73, 123, "タイガー", 50, 4),
    "skuba": CharacterStats("skuba", 0.31, 0.17, 105, 0.71, 100, "スモーカー", 50, 6),
    "C0M": CharacterStats("C0M", 0.28, 0.16, 120, 0.69, 95, "シーカー", 60, 9),
    "mada": CharacterStats("mada", 0.34, 0.23, 105, 0.74, 105, "フラッシュ", 55, 4),
    "s0m": CharacterStats("s0m", 0.3, 0.23, 145, 0.7, 116, "スモーカー", 55, 7),
    "Lohen": CharacterStats("Lohen", 0.4, 0.31, 70, 0.75, 130, "タイガー", 100, 10),
    "Furina": CharacterStats("Furina", 0.29, 0.09, 149, 0.69, 129, "フラッシュ", 20, 0),
    "Lisa": CharacterStats("Lisa", 0.36, 0.25, 120, 0.78, 99, "スモーカー", 40, 5),
    "Jean": CharacterStats("Jean", 0.35, 0.2, 115, 0.75, 135, "シーカー", 40, 5),
    "Kachina": CharacterStats("Kachina", 0.5, 0.2, 70, 0.9, 100, "タイガー", 40, 3),
    "IbarakiNinja": CharacterStats(
        "IbarakiNinja", 0.28, 0.16, 110, 0.75, 115, "シーカー", 50, 10
    ),
    "Canezerra": CharacterStats(
        "Canezerra", 0.45, 0.17, 85, 0.8, 147, "タイガー", 80, 7
    ),
    "Arlecchino": CharacterStats(
        "Arlecchino", 0.47, 0.19, 135, 0.6, 125, "タイガー", 100, 1
    ),
    "WsLeo": CharacterStats("WsLeo", 0.36, 0.21, 130, 0.78, 80, "シーカー", 50, 9),
    "Zest": CharacterStats("Zest", 0.5, 0.25, 75, 0.76, 121, "シーカー", 100, 9),
    "Tartaglia": CharacterStats(
        "Tartaglia", 0.55, 0.15, 75, 0.72, 150, "フラッシュ", 80, 7
    ),
    "Nanasaki": CharacterStats(
        "Nanasaki", 0.5, 0.2, 120, 0.65, 80, "スモーカー", 70, 8
    ),
    "Less": CharacterStats("Less", 0.35, 0.22, 86, 0.74, 104, "タイガー", 80, 3),
    "WoohyuN": CharacterStats("WoohyuN", 0.46, 0.2, 85, 0.71, 150, "タイガー", 55, 8),
    "Asuna": CharacterStats("Asuna", 0.29, 0.17, 85, 0.75, 125, "フラッシュ", 65, 7),
    "Cryocells": CharacterStats(
        "Cryocells", 0.29, 0.18, 85, 0.76, 110, "タイガー", 65, 6
    ),
    "vo0kashu": CharacterStats(
        "vo0kashu", 0.35, 0.23, 80, 0.76, 115, "タイガー", 50, 4
    ),
    "Absol": CharacterStats("Absol", 0.15, 0.31, 130, 0.87, 155, "タイガー", 50, 10),
    "marteen": CharacterStats("marteen", 0.39, 0.2, 80, 0.78, 120, "タイガー", 85, 4),
    "Crewn": CharacterStats("Crewn", 0.25, 0.25, 135, 0.75, 115, "スモーカー", 65, 5),
    "Loita": CharacterStats("Loita", 0.33, 0.22, 111, 0.77, 111, "スモーカー", 77, 6),
    "cNed": CharacterStats("cNed", 0.41, 0.12, 85, 0.95, 155, "タイガー", 180, 0),
    "soulcas": CharacterStats(
        "soulcas", 0.53, 0.17, 120, 0.63, 155, "フラッシュ", 90, 0
    ),
    "trexx": CharacterStats("trexx", 0.25, 0.2, 100, 0.85, 135, "シーカー", 30, 0),
    "mindfreak": CharacterStats(
        "mindfreak", 0.3, 0.3, 140, 0.7, 100, "スモーカー", 50, 8
    ),
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
