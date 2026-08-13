from dataclasses import dataclass, replace
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
    "Chronicle": CharacterStats(
        "Chronicle", 0.35, 0.23, 105, 0.78, 135, "フラッシュ", 80, 3, 9
    ),
    "Demon1": CharacterStats(
        "Demon1", 0.5, 0.13, 78, 0.85, 145, "スモーカー", 150, 4, 7
    ),
    "something": CharacterStats(
        "something", 0.35, 0.26, 65, 0.88, 165, "タイガー", 85, 7, 2
    ),
    "Leo": CharacterStats("Leo", 0.27, 0.22, 150, 0.83, 135, "シーカー", 65, 1, 10),
    "Alfajer": CharacterStats(
        "Alfajer", 0.45, 0.13, 60, 0.91, 145, "タイガー", 60, 4, 7
    ),
    "Derke": CharacterStats("Derke", 0.35, 0.24, 90, 0.75, 115, "タイガー", 90, 8, 8),
    "Boaster": CharacterStats(
        "Boaster", 0.2, 0.12, 175, 0.62, 96, "スモーカー", 50, 5, 10
    ),
    "Aspas": CharacterStats("Aspas", 0.4, 0.24, 90, 0.87, 120, "タイガー", 150, 10, 10),
    "F0rsakeN": CharacterStats(
        "F0rsakeN", 0.27, 0.14, 125, 0.69, 95, "スモーカー", 50, 3, 6
    ),
    "Jinggg": CharacterStats("Jinggg", 0.3, 0.26, 68, 0.71, 90, "タイガー", 50, 4, 3),
    "d4v41": CharacterStats("d4v41", 0.35, 0.17, 70, 0.71, 105, "シーカー", 40, 5, 3),
    "Sato": CharacterStats("Sato", 0.4, 0.19, 67, 0.73, 105, "フラッシュ", 60, 6, 7),
    "jawgemo": CharacterStats(
        "jawgemo", 0.28, 0.31, 93, 0.67, 135, "タイガー", 75, 8, 9
    ),
    "valyn": CharacterStats("valyn", 0.26, 0.13, 140, 0.64, 95, "スモーカー", 40, 5, 5),
    "Ethan": CharacterStats("Ethan", 0.3, 0.18, 150, 0.8, 90, "シーカー", 55, 4, 10),
    "Wo0t": CharacterStats("Wo0t", 0.4, 0.16, 78, 0.75, 120, "タイガー", 75, 9, 7),
    "kaajak": CharacterStats(
        "kaajak", 0.35, 0.2, 115, 0.76, 100, "フラッシュ", 70, 7, 6
    ),
    "Meiy": CharacterStats("Meiy", 0.35, 0.22, 70, 0.73, 110, "タイガー", 90, 7, 4),
    "Verno": CharacterStats("Verno", 0.35, 0.16, 86, 0.7, 95, "シーカー", 80, 7, 6),
    "Sayonara": CharacterStats(
        "Sayonara", 0.33, 0.19, 95, 0.7, 90, "シーカー", 80, 5, 10
    ),
    "Jamppi": CharacterStats(
        "Jamppi", 0.27, 0.15, 125, 0.68, 85, "フラッシュ", 55, 3, 5
    ),
    "Boostio": CharacterStats(
        "Boostio", 0.28, 0.15, 128, 0.68, 85, "スモーカー", 70, 6, 9
    ),
    "Primmie": CharacterStats(
        "Primmie", 0.47, 0.24, 59, 0.79, 98, "タイガー", 90, 5, 2
    ),
    "まーやまくん": CharacterStats(
        "まーやまくん", 0.75, 0.21, 90, 0.8, 140, "シーカー", 50, 1, 7
    ),
    "おもこ": CharacterStats(
        "おもこ", 0.67, 0.67, 67, 0.67, 67, "フラッシュ", 67, 0, 8
    ),
    "Meteor": CharacterStats(
        "Meteor", 0.37, 0.29, 73, 0.78, 95, "フラッシュ", 75, 8, 9
    ),
    "Laz": CharacterStats("Laz", 0.4, 0.14, 70, 0.75, 86, "フラッシュ", 80, 6, 8),
    "ZMJKK": CharacterStats("ZMJKK", 0.3, 0.17, 65, 0.71, 150, "タイガー", 90, 9, 8),
    "Brawk": CharacterStats("Brawk", 0.32, 0.15, 90, 0.69, 95, "シーカー", 90, 6, 8),
    "HYUNMIN": CharacterStats(
        "HYUNMIN", 0.35, 0.24, 76, 0.75, 110, "フラッシュ", 70, 5, 6
    ),
    "Flashback": CharacterStats(
        "Flashback", 0.38, 0.22, 79, 0.77, 130, "フラッシュ", 70, 5, 3
    ),
    "Mako": CharacterStats("Mako", 0.28, 0.16, 120, 0.7, 85, "スモーカー", 50, 5, 5),
    "t3xture": CharacterStats(
        "t3xture", 0.33, 0.19, 93, 0.75, 150, "タイガー", 75, 7, 8
    ),
    "trent": CharacterStats("trent", 0.3, 0.18, 120, 0.71, 85, "シーカー", 40, 5, 5),
    "leaf": CharacterStats("leaf", 0.38, 0.17, 89, 0.74, 80, "フラッシュ", 40, 5, 3),
    "keiko": CharacterStats("keiko", 0.31, 0.19, 95, 0.72, 90, "スモーカー", 70, 8, 7),
    "Rb": CharacterStats("Rb", 0.29, 0.23, 110, 0.69, 80, "スモーカー", 40, 4, 5),
    "stax": CharacterStats("stax", 0.29, 0.15, 130, 0.69, 80, "フラッシュ", 70, 4, 5),
    "tex": CharacterStats("tex", 0.35, 0.19, 85, 0.74, 95, "フラッシュ", 60, 7, 8),
    "Mazino": CharacterStats(
        "Mazino", 0.33, 0.18, 100, 0.72, 94, "スモーカー", 60, 7, 8
    ),
    "Zekken": CharacterStats(
        "Zekken", 0.35, 0.25, 85, 0.7, 105, "フラッシュ", 100, 8, 8
    ),
    "BABYBAY": CharacterStats(
        "BABYBAY", 0.4, 0.15, 85, 0.73, 100, "タイガー", 55, 7, 5
    ),
    "Dep": CharacterStats("Dep", 0.33, 0.22, 75, 0.73, 100, "タイガー", 50, 5, 6),
    "SugarZ3ro": CharacterStats(
        "SugarZ3ro", 0.29, 0.18, 120, 0.67, 93, "スモーカー", 40, 5, 5
    ),
    "Buzz": CharacterStats("Buzz", 0.3, 0.23, 80, 0.72, 100, "タイガー", 60, 5, 5),
    "TenZ": CharacterStats("TenZ", 0.38, 0.23, 103, 0.77, 137, "スモーカー", 170, 9, 9),
    "eggsterr": CharacterStats(
        "eggsterr", 0.3, 0.22, 88, 0.7, 95, "フラッシュ", 50, 10, 3
    ),
    "Rossy": CharacterStats("Rossy", 0.31, 0.18, 89, 0.7, 100, "シーカー", 50, 9, 2),
    "Rarga": CharacterStats("Rarga", 0.35, 0.19, 75, 0.75, 120, "タイガー", 50, 7, 2),
    "Lysoar": CharacterStats(
        "Lysoar", 0.33, 0.17, 105, 0.71, 100, "スモーカー", 50, 7, 2
    ),
    "Smoggy": CharacterStats(
        "Smoggy", 0.38, 0.19, 75, 0.73, 120, "スモーカー", 50, 8, 6
    ),
    "CHICHOO": CharacterStats(
        "CHICHOO", 0.32, 0.18, 80, 0.74, 100, "スモーカー", 50, 9, 8
    ),
    "crashies": CharacterStats(
        "crashies", 0.28, 0.17, 140, 0.68, 80, "シーカー", 45, 0, 3
    ),
    "FNS": CharacterStats("FNS", 0.17, 0.13, 190, 0.63, 70, "スモーカー", 10, 0, 7),
    "nAts": CharacterStats("nAts", 0.29, 0.18, 140, 0.72, 88, "スモーカー", 50, 9, 9),
    "Lar0k": CharacterStats("Lar0k", 0.34, 0.18, 85, 0.73, 123, "タイガー", 50, 4, 5),
    "skuba": CharacterStats(
        "skuba", 0.31, 0.17, 105, 0.71, 100, "スモーカー", 50, 6, 8
    ),
    "C0M": CharacterStats("C0M", 0.28, 0.16, 120, 0.69, 95, "シーカー", 60, 9, 7),
    "mada": CharacterStats("mada", 0.34, 0.23, 105, 0.74, 105, "フラッシュ", 55, 4, 7),
    "s0m": CharacterStats("s0m", 0.3, 0.23, 145, 0.7, 116, "スモーカー", 55, 7, 6),
    "Lohen": CharacterStats("Lohen", 0.4, 0.31, 70, 0.75, 130, "タイガー", 100, 10, 10),
    "Furina": CharacterStats(
        "Furina", 0.29, 0.09, 149, 0.69, 129, "フラッシュ", 20, 1, 0
    ),
    "Lisa": CharacterStats("Lisa", 0.36, 0.25, 120, 0.78, 99, "スモーカー", 40, 5, 7),
    "Jean": CharacterStats("Jean", 0.35, 0.2, 115, 0.75, 135, "シーカー", 40, 5, 7),
    "Kachina": CharacterStats("Kachina", 0.5, 0.2, 70, 0.9, 100, "タイガー", 40, 3, 3),
    "IbarakiNinja": CharacterStats(
        "IbarakiNinja", 0.28, 0.16, 110, 0.75, 115, "シーカー", 50, 10, 5
    ),
    "Canezerra": CharacterStats(
        "Canezerra", 0.45, 0.17, 85, 0.8, 147, "タイガー", 90, 7, 6
    ),
    "Arlecchino": CharacterStats(
        "Arlecchino", 0.47, 0.19, 135, 0.6, 125, "タイガー", 100, 1, 8
    ),
    "WsLeo": CharacterStats("WsLeo", 0.36, 0.21, 130, 0.78, 80, "シーカー", 50, 9, 8),
    "Zest": CharacterStats("Zest", 0.42, 0.25, 75, 0.77, 121, "シーカー", 100, 9, 5),
    "Tartaglia": CharacterStats(
        "Tartaglia", 0.55, 0.15, 75, 0.72, 150, "フラッシュ", 80, 7, 5
    ),
    "Nanasaki": CharacterStats(
        "Nanasaki", 0.35, 0.2, 120, 0.65, 80, "スモーカー", 70, 8, 3
    ),
    "Less": CharacterStats("Less", 0.35, 0.22, 86, 0.74, 104, "タイガー", 80, 3, 6),
    "WoohyuN": CharacterStats(
        "WoohyuN", 0.46, 0.2, 85, 0.71, 150, "タイガー", 55, 8, 4
    ),
    "Asuna": CharacterStats("Asuna", 0.29, 0.17, 85, 0.75, 125, "フラッシュ", 65, 7, 8),
    "Cryocells": CharacterStats(
        "Cryocells", 0.29, 0.18, 85, 0.76, 110, "タイガー", 65, 6, 8
    ),
    "vo0kashu": CharacterStats(
        "vo0kashu", 0.35, 0.23, 80, 0.76, 115, "タイガー", 50, 4, 5
    ),
    "Absol": CharacterStats("Absol", 0.15, 0.31, 130, 0.87, 155, "タイガー", 50, 10, 3),
    "marteen": CharacterStats(
        "marteen", 0.39, 0.2, 80, 0.78, 120, "タイガー", 85, 4, 6
    ),
    "Crewn": CharacterStats("Crewn", 0.24, 0.25, 95, 0.74, 88, "スモーカー", 65, 5, 6),
    "Loita": CharacterStats("Loita", 0.26, 0.22, 100, 0.76, 95, "スモーカー", 77, 6, 6),
    "Katarina": CharacterStats(
        "Katarina", 0.41, 0.18, 86, 0.81, 170, "スモーカー", 110, 10, 6
    ),
    "SereNa": CharacterStats(
        "SereNa", 0.29, 0.19, 119, 0.9, 139, "シーカー", 120, 10, 6
    ),
    "cNed": CharacterStats("cNed", 0.41, 0.12, 85, 0.95, 155, "タイガー", 180, 0, 8),
    "soulcas": CharacterStats(
        "soulcas", 0.53, 0.17, 120, 0.63, 155, "フラッシュ", 90, 0, 8
    ),
    "trexx": CharacterStats("trexx", 0.25, 0.2, 100, 0.85, 135, "シーカー", 30, 0, 8),
    "mindfreak": CharacterStats(
        "mindfreak", 0.25, 0.3, 145, 0.75, 100, "スモーカー", 50, 8, 10
    ),
    "Rosé": CharacterStats("Rosé", 0.24, 0.19, 130, 0.76, 89, "シーカー", 45, 3, 7),
    "lovers rock": CharacterStats(
        "lovers rock", 0.29, 0.2, 95, 0.77, 90, "タイガー", 55, 6, 6
    ),
    "Tortlilyan": CharacterStats(
        "Tortlilyan", 0.23, 0.39, 123, 0.9, 156, "タイガー", 50, 10, 4
    ),
    "ろびぃな": CharacterStats(
        "ろびぃな", 0.25, 0.55, 80, 0.65, 122, "スモーカー", 50, 5, 8
    ),
    "えんぺん": CharacterStats(
        "えんぺん", 0.5, 0.17, 185, 0.75, 125, "フラッシュ", 65, 1, 8
    ),
    "いぐるん": CharacterStats(
        "いぐるん", 0.33, 0.22, 75, 0.77, 131, "シーカー", 55, 4, 8
    ),
    "夢の街": CharacterStats(
        "夢の街", 0.35, 0.23, 80, 0.76, 115, "フラッシュ", 50, 3, 8
    ),
    "Retloff": CharacterStats(
        "Retloff", 0.23, 0.2, 158, 0.69, 98, "フラッシュ", 50, 3, 3
    ),
    "alecks": CharacterStats(
        "alecks", 0.38, 0.13, 145, 0.7, 115, "タイガー", 115, 9, 9
    ),
    "icy": CharacterStats("icy", 0.26, 0.21, 75, 0.78, 95, "タイガー", 35, 5, 9),
    "yay": CharacterStats("yay", 0.41, 0.14, 70, 0.8, 100, "フラッシュ", 90, 9, 4),
    "eeiu": CharacterStats("eeiu", 0.27, 0.22, 85, 0.76, 100, "シーカー", 45, 8, 7),
    "Munchkin": CharacterStats(
        "Munchkin", 0.28, 0.17, 135, 0.75, 85, "シーカー", 75, 8, 3
    ),
    "Foxy9": CharacterStats("Foxy9", 0.31, 0.18, 85, 0.79, 100, "タイガー", 65, 5, 4),
    "benjyfishy": CharacterStats(
        "benjyfishy", 0.34, 0.18, 95, 0.78, 150, "スモーカー", 70, 9, 7
    ),
    "RieNs": CharacterStats("RieNs", 0.25, 0.16, 125, 0.76, 100, "シーカー", 75, 3, 8),
    "Kr1stal": CharacterStats(
        "Kr1stal", 0.4, 0.16, 115, 0.81, 120, "シーカー", 80, 8, 5
    ),
    "Xdll": CharacterStats("Xdll", 0.23, 0.24, 125, 0.78, 140, "シーカー", 45, 7, 4),
    "SyouTa": CharacterStats("SyouTa", 0.45, 0.2, 90, 0.75, 140, "タイガー", 95, 8, 9),
    "S1Mon": CharacterStats(
        "S1Mon", 0.35, 0.15, 95, 0.8, 135, "フラッシュ", 110, 10, 10
    ),
    "koldamenta": CharacterStats(
        "koldamenta", 0.2, 0.25, 105, 0.75, 115, "シーカー", 50, 9, 7
    ),
    "PatMen": CharacterStats("PatMen", 0.33, 0.2, 80, 0.8, 125, "スモーカー", 55, 5, 8),
    "eKo": CharacterStats("eKo", 0.3, 0.21, 85, 0.76, 110, "タイガー", 60, 5, 8),
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
