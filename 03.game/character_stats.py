from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass(frozen=True)
class CharacterStats:
    name: str
    hs_pct: float
    dodge_pct: float
    iq: int
    hit_pct: float
    role: str

CHARACTER_TABLE: Dict[str, CharacterStats] = {
    'Chronicle': CharacterStats('Chronicle', 0.35, 0.2, 105, 0.7, 'フラッシュ'),
    'Demon1': CharacterStats('Demon1', 0.45, 0.13, 85, 0.76, 'スモーカー'),
    'something': CharacterStats('something', 0.35, 0.24, 70, 0.77, 'タイガー'),
    'Leo': CharacterStats('Leo', 0.27, 0.22, 150, 0.83, 'シーカー'),
    'Alfajer': CharacterStats('Alfajer', 0.45, 0.12, 65, 0.8, 'タイガー'),
    'Derke': CharacterStats('Derke', 0.35, 0.23, 95, 0.7, 'フラッシュ'),
    'Boaster': CharacterStats('Boaster', 0.2, 0.12, 165, 0.62, 'スモーカー'),
    'Aspas': CharacterStats('Aspas', 0.4, 0.24, 90, 0.75, 'タイガー'),
    'F0rsakeN': CharacterStats('F0rsakeN', 0.26, 0.14, 125, 0.69, 'スモーカー'),
    'Jinggg': CharacterStats('Jinggg', 0.3, 0.23, 68, 0.71, 'タイガー'),
    'd4v41': CharacterStats('d4v41', 0.35, 0.17, 70, 0.71, 'シーカー'),
    'Sato': CharacterStats('Sato', 0.4, 0.19, 67, 0.73, 'フラッシュ'),
    'jawgemo': CharacterStats('jawgemo', 0.28, 0.29, 93, 0.67, 'タイガー'),
    'valyn': CharacterStats('valyn', 0.25, 0.13, 135, 0.64, 'スモーカー'),
    'Ethan': CharacterStats('Ethan', 0.3, 0.18, 140, 0.8, 'シーカー'),
    'Wo0t': CharacterStats('Wo0t', 0.4, 0.16, 78, 0.75, 'タイガー'),
    'kaajak': CharacterStats('kaajak', 0.35, 0.2, 95, 0.76, 'フラッシュ'),
    'Meiy': CharacterStats('Meiy', 0.35, 0.22, 70, 0.73, 'タイガー'),
    'Verno': CharacterStats('Verno', 0.35, 0.16, 86, 0.7, 'シーカー'),
    'Sayonara': CharacterStats('Sayonara', 0.33, 0.19, 89, 0.7, 'シーカー'),
    'Jamppi': CharacterStats('Jamppi', 0.25, 0.15, 125, 0.68, 'フラッシュ'),
    'Boostio': CharacterStats('Boostio', 0.28, 0.15, 128, 0.68, 'スモーカー'),
    'Primmie': CharacterStats('Primmie', 0.47, 0.24, 59, 0.75, 'タイガー'),
    'Tortlilyan': CharacterStats('Tortlilyan', 0.2, 0.34, 123, 0.76, 'タイガー'),
    'まーやまくん': CharacterStats('まーやまくん', 0.6, 0.22, 79, 0.67, 'シーカー'),
    'おもこ': CharacterStats('おもこ', 0.067, 0.67, 67, 0.67, 'フラッシュ'),
    'Meteor': CharacterStats('Meteor', 0.36, 0.28, 73, 0.76, 'フラッシュ'),
    'Laz': CharacterStats('Laz', 0.4, 0.14, 70, 0.71, 'フラッシュ'),
    'ZMJKK': CharacterStats('ZMJKK', 0.3, 0.17, 55, 0.71, 'タイガー'),
    'Brawk': CharacterStats('Brawk', 0.3, 0.14, 79, 0.67, 'シーカー'),
    'HYUNMIN': CharacterStats('HYUNMIN', 0.35, 0.21, 76, 0.75, 'フラッシュ'),
    'Flashback': CharacterStats('Flashback', 0.38, 0.21, 79, 0.76, 'フラッシュ'),
    'Mako': CharacterStats('Mako', 0.28, 0.16, 120, 0.7, 'スモーカー'),
    't3xture': CharacterStats('t3xture', 0.33, 0.19, 93, 0.75, 'タイガー'),
    'trent': CharacterStats('trent', 0.3, 0.18, 120, 0.71, 'シーカー'),
    'leaf': CharacterStats('leaf', 0.35, 0.17, 89, 0.74, 'フラッシュ'),
    'keiko': CharacterStats('keiko', 0.31, 0.19, 89, 0.72, 'スモーカー'),
    'Rb': CharacterStats('Rb', 0.29, 0.18, 88, 0.67, 'スモーカー'),
    'stax': CharacterStats('stax', 0.29, 0.15, 99, 0.67, 'フラッシュ'),
    'tex': CharacterStats('tex', 0.34, 0.18, 85, 0.73, 'フラッシュ'),
    'Mazino': CharacterStats('Mazino', 0.33, 0.18, 100, 0.73, 'スモーカー'),
    'Zekken': CharacterStats('Zekken', 0.35, 0.22, 85, 0.7, 'フラッシュ'),
    'BABYBAY': CharacterStats('BABYBAY', 0.4, 0.15, 85, 0.73, 'タイガー'),
    'Dep': CharacterStats('Dep', 0.33, 0.22, 75, 0.73, 'タイガー'),
    'SugarZ3ro': CharacterStats('SugarZ3ro', 0.29, 0.18, 115, 0.65, 'スモーカー'),
    'Buzz': CharacterStats('Buzz', 0.3, 0.22, 80, 0.72, 'フラッシュ'),
    'TenZ': CharacterStats('TenZ', 0.35, 0.22, 85, 0.76, 'スモーカー'),
    'eggsterr': CharacterStats('eggsterr', 0.3, 0.22, 88, 0.7, 'フラッシュ'),
    'Rossy': CharacterStats('Rossy', 0.3, 0.18, 89, 0.7, 'シーカー'),
    'Rarga': CharacterStats('Rarga', 0.35, 0.19, 75, 0.75, 'タイガー'),
    'Lysoar': CharacterStats('Lysoar', 0.3, 0.17, 105, 0.7, 'スモーカー'),
    'Smoggy': CharacterStats('Smoggy', 0.38, 0.19, 75, 0.73, 'スモーカー'),
    'CHICHOO': CharacterStats('CHICHOO', 0.32, 0.18, 80, 0.74, 'スモーカー'),
    'crashies': CharacterStats('crashies', 0.28, 0.17, 135, 0.68, 'シーカー'),
    'FNS': CharacterStats('FNS', 0.15, 0.13, 170, 0.63, 'スモーカー'),
    'nAts': CharacterStats('nAts', 0.29, 0.18, 125, 0.71, 'スモーカー'),
    'Lar0k': CharacterStats('Lar0k', 0.34, 0.18, 85, 0.73, 'タイガー'),
    'skuba': CharacterStats('skuba', 0.31, 0.17, 95, 0.7, 'スモーカー'),
    'C0M': CharacterStats('C0M', 0.28, 0.16, 120, 0.69, 'シーカー'),
    'mada': CharacterStats('mada', 0.34, 0.23, 105, 0.74, 'フラッシュ'),
    's0m': CharacterStats('s0m', 0.3, 0.23, 135, 0.7, 'スモーカー'),
    # 追加したキャラクター
    'Lisa': CharacterStats('Lisa', 0.36, 0.25, 141, 0.78, 'スモーカー'),
    'Jean': CharacterStats('Jean', 0.3, 0.3, 145, 0.75, 'シーカー'),
    'Lohen': CharacterStats('Lohen', 0.4, 0.35, 75, 0.75, 'タイガー'),
    'Furina': CharacterStats('Furina', 0.29, 0.19, 140, 0.78, 'フラッシュ'),
    'Kachina': CharacterStats('Kachina', 0.5, 0.2, 69, 0.9, 'タイガー'),
}

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