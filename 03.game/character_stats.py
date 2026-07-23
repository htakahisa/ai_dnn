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

CHARACTER_TABLE: Dict[str, CharacterStats] = {
    'Chronicle': CharacterStats('Chronicle', 0.35, 0.2, 105, 0.7, 135, 'フラッシュ', 80),
    'Demon1': CharacterStats('Demon1', 0.5, 0.13, 80, 0.86, 145, 'スモーカー', 150),
    'something': CharacterStats('something', 0.35, 0.24, 70, 0.88, 165, 'タイガー', 60),
    'Leo': CharacterStats('Leo', 0.27, 0.22, 150, 0.83, 135, 'シーカー', 65),
    'Alfajer': CharacterStats('Alfajer', 0.45, 0.13, 60, 0.91, 145, 'タイガー', 60),
    'Derke': CharacterStats('Derke', 0.35, 0.24, 90, 0.75, 115, 'フラッシュ', 100),
    'Boaster': CharacterStats('Boaster', 0.2, 0.12, 165, 0.62, 96, 'スモーカー', 50),
    'Aspas': CharacterStats('Aspas', 0.4, 0.24, 90, 0.87, 120, 'タイガー', 130),
    'F0rsakeN': CharacterStats('F0rsakeN', 0.26, 0.14, 125, 0.69, 95, 'スモーカー', 50),
    'Jinggg': CharacterStats('Jinggg', 0.3, 0.26, 68, 0.71, 90, 'タイガー', 50),
    'd4v41': CharacterStats('d4v41', 0.35, 0.17, 70, 0.71, 105, 'シーカー', 40),
    'Sato': CharacterStats('Sato', 0.4, 0.19, 67, 0.73, 105, 'フラッシュ', 60),
    'jawgemo': CharacterStats('jawgemo', 0.28, 0.31, 93, 0.67, 135, 'タイガー', 75),
    'valyn': CharacterStats('valyn', 0.26, 0.13, 140, 0.64, 95, 'スモーカー', 40),
    'Ethan': CharacterStats('Ethan', 0.3, 0.18, 150, 0.8, 90, 'シーカー', 55),
    'Wo0t': CharacterStats('Wo0t', 0.4, 0.16, 78, 0.75, 120, 'タイガー', 75),
    'kaajak': CharacterStats('kaajak', 0.35, 0.2, 95, 0.76, 100, 'フラッシュ', 70),
    'Meiy': CharacterStats('Meiy', 0.35, 0.22, 70, 0.73, 110, 'タイガー', 90),
    'Verno': CharacterStats('Verno', 0.35, 0.16, 86, 0.7, 95, 'シーカー', 80),
    'Sayonara': CharacterStats('Sayonara', 0.33, 0.19, 89, 0.7, 90, 'シーカー', 80),
    'Jamppi': CharacterStats('Jamppi', 0.27, 0.15, 125, 0.68, 85, 'フラッシュ', 55),
    'Boostio': CharacterStats('Boostio', 0.28, 0.15, 128, 0.68, 85, 'スモーカー', 70),
    'Primmie': CharacterStats('Primmie', 0.47, 0.24, 59, 0.75, 98, 'タイガー', 90),
    'Tortlilyan': CharacterStats('Tortlilyan', 0.23, 0.34, 123, 0.81, 155, 'タイガー', 50),
    'まーやまくん': CharacterStats('まーやまくん', 0.7, 0.21, 90, 0.67, 140, 'シーカー', 50),
    'おもこ': CharacterStats('おもこ', 0.067, 0.67, 67, 0.67, 67, 'フラッシュ', 50),
    'Meteor': CharacterStats('Meteor', 0.36, 0.28, 73, 0.76, 95, 'フラッシュ', 75),
    'Laz': CharacterStats('Laz', 0.4, 0.14, 70, 0.75, 86, 'フラッシュ', 80),
    'ZMJKK': CharacterStats('ZMJKK', 0.3, 0.17, 65, 0.71, 150, 'タイガー', 90),
    'Brawk': CharacterStats('Brawk', 0.32, 0.15, 90, 0.69, 95, 'シーカー', 90),
    'HYUNMIN': CharacterStats('HYUNMIN', 0.35, 0.24, 76, 0.75, 110, 'フラッシュ', 70),
    'Flashback': CharacterStats('Flashback', 0.38, 0.22, 79, 0.77, 130, 'フラッシュ', 70),
    'Mako': CharacterStats('Mako', 0.28, 0.16, 120, 0.7, 85, 'スモーカー', 50),
    't3xture': CharacterStats('t3xture', 0.33, 0.19, 93, 0.75, 150, 'タイガー', 75),
    'trent': CharacterStats('trent', 0.3, 0.18, 120, 0.71, 85, 'シーカー', 40),
    'leaf': CharacterStats('leaf', 0.38, 0.17, 89, 0.74, 80, 'フラッシュ', 40),
    'keiko': CharacterStats('keiko', 0.31, 0.19, 89, 0.72, 90, 'スモーカー', 70),
    'Rb': CharacterStats('Rb', 0.29, 0.23, 110, 0.67, 80, 'スモーカー', 40),
    'stax': CharacterStats('stax', 0.29, 0.15, 130, 0.67, 80, 'フラッシュ', 70),
    'tex': CharacterStats('tex', 0.34, 0.18, 85, 0.73, 95, 'フラッシュ', 60),
    'Mazino': CharacterStats('Mazino', 0.33, 0.18, 100, 0.73, 93, 'スモーカー', 60),
    'Zekken': CharacterStats('Zekken', 0.35, 0.25, 85, 0.7, 105, 'フラッシュ', 100),
    'BABYBAY': CharacterStats('BABYBAY', 0.4, 0.15, 85, 0.73, 100, 'タイガー', 55),
    'Dep': CharacterStats('Dep', 0.33, 0.22, 75, 0.73, 100, 'タイガー', 50),
    'SugarZ3ro': CharacterStats('SugarZ3ro', 0.29, 0.18, 120, 0.65, 93, 'スモーカー', 40),
    'Buzz': CharacterStats('Buzz', 0.3, 0.23, 80, 0.72, 100, 'タイガー', 60),
    'TenZ': CharacterStats('TenZ', 0.38, 0.23, 105, 0.76, 120, 'スモーカー', 170),
    'eggsterr': CharacterStats('eggsterr', 0.3, 0.22, 88, 0.7, 95, 'フラッシュ', 50),
    'Rossy': CharacterStats('Rossy', 0.3, 0.18, 89, 0.7, 100, 'シーカー', 50),
    'Rarga': CharacterStats('Rarga', 0.35, 0.19, 75, 0.75, 120, 'タイガー', 50),
    'Lysoar': CharacterStats('Lysoar', 0.33, 0.17, 105, 0.71, 100, 'スモーカー', 50),
    'Smoggy': CharacterStats('Smoggy', 0.38, 0.19, 75, 0.73, 120, 'スモーカー', 50),
    'CHICHOO': CharacterStats('CHICHOO', 0.32, 0.18, 80, 0.74, 100, 'スモーカー', 50),
    'crashies': CharacterStats('crashies', 0.28, 0.17, 140, 0.68, 80, 'シーカー', 45),
    'FNS': CharacterStats('FNS', 0.15, 0.13, 170, 0.63, 70, 'スモーカー', 40),
    'nAts': CharacterStats('nAts', 0.29, 0.18, 125, 0.71, 88, 'スモーカー', 50),
    'Lar0k': CharacterStats('Lar0k', 0.34, 0.18, 85, 0.73, 123, 'タイガー', 50),
    'skuba': CharacterStats('skuba', 0.31, 0.17, 95, 0.71, 100, 'スモーカー', 50),
    'C0M': CharacterStats('C0M', 0.28, 0.16, 120, 0.69, 95, 'シーカー', 60),
    'mada': CharacterStats('mada', 0.34, 0.23, 105, 0.74, 105, 'フラッシュ', 55),
    's0m': CharacterStats('s0m', 0.3, 0.23, 145, 0.7, 116, 'スモーカー', 55),
    'Lohen': CharacterStats('Lohen', 0.4, 0.31, 70, 0.75, 130, 'タイガー', 100),
    'Furina': CharacterStats('Furina', 0.29, 0.19, 149, 0.69, 129, 'フラッシュ', 30),
    'Lisa': CharacterStats('Lisa', 0.36, 0.25, 120, 0.78, 99, 'スモーカー', 80),
    'Jean': CharacterStats('Jean', 0.35, 0.2, 115, 0.75, 135, 'シーカー', 50),
    'Kachina': CharacterStats('Kachina', 0.45, 0.2, 60, 0.8, 90, 'タイガー', 40),
    'IbarakiNinja': CharacterStats('IbarakiNinja', 0.28, 0.16, 110, 0.75, 115, 'シーカー', 50),
    'Canezera': CharacterStats('Canezera', 0.41, 0.17, 95, 0.77, 135, 'タイガー', 80),
    'Arlecchino': CharacterStats('Arlecchino', 0.47, 0.19, 135, 0.6, 125, 'タイガー', 100),
    'WsLeo': CharacterStats('WsLeo', 0.36, 0.21, 130, 0.78, 80, 'シーカー', 50),
    'Zest': CharacterStats('Zest', 0.5, 0.25, 75, 0.76, 121, 'シーカー', 100),
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
