from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass(frozen=True)
class CharacterStats:
    name: str
    hs_pct: float
    dodge_pct: float
    iq: int
    hit_pct: float
    reaction_speed: float
    role: str

CHARACTER_TABLE: Dict[str, CharacterStats] = {
    'Tortlilyan': CharacterStats('Tortlilyan', 0.23, 0.34, 123, 0.81, 155.0, 'タイガー'),
    'まーやまくん': CharacterStats('まーやまくん', 0.7, 0.21, 90, 0.67, 140.0, 'シーカー'),
    'Leo': CharacterStats('Leo', 0.27, 0.22, 150, 0.83, 135.0, 'シーカー'),
    'something': CharacterStats('something', 0.35, 0.24, 75, 0.88, 165.0, 'タイガー'),
    'Demon1': CharacterStats('Demon1', 0.5, 0.13, 85, 0.88, 145.0, 'スモーカー'),
    'Aspas': CharacterStats('Aspas', 0.4, 0.24, 90, 0.87, 120.0, 'タイガー'),
    'Alfajer': CharacterStats('Alfajer', 0.45, 0.13, 65, 0.91, 145.0, 'タイガー'),
    'Lohen': CharacterStats('Lohen', 0.4, 0.31, 70, 0.75, 130.0, 'タイガー'),
    'Jean': CharacterStats('Jean', 0.35, 0.2, 115, 0.75, 135.0, 'シーカー'),
    'Arlecchino': CharacterStats('Arlecchino', 0.47, 0.19, 135, 0.6, 125.0, 'タイガー'),
    'Lisa': CharacterStats('Lisa', 0.36, 0.25, 120, 0.78, 99.0, 'スモーカー'),
    's0m': CharacterStats('s0m', 0.3, 0.23, 145, 0.7, 116.0, 'スモーカー'),
    'Furina': CharacterStats('Furina', 0.29, 0.19, 149, 0.69, 129.0, 'フラッシュ'),
    'Canezera': CharacterStats('Canezera', 0.41, 0.17, 95, 0.77, 135.0, 'タイガー'),
    'WsLeo': CharacterStats('WsLeo', 0.39, 0.21, 130, 0.8, 75.0, 'シーカー'),
    't3xture': CharacterStats('t3xture', 0.33, 0.19, 93, 0.75, 150.0, 'タイガー'),
    'Flashback': CharacterStats('Flashback', 0.38, 0.22, 79, 0.77, 130.0, 'フラッシュ'),
    'Ethan': CharacterStats('Ethan', 0.3, 0.18, 140, 0.8, 90.0, 'シーカー'),
    'Chronicle': CharacterStats('Chronicle', 0.35, 0.2, 105, 0.7, 135.0, 'フラッシュ'),
    'jawgemo': CharacterStats('jawgemo', 0.28, 0.29, 93, 0.67, 135.0, 'タイガー'),
    'mada': CharacterStats('mada', 0.34, 0.23, 105, 0.74, 105.0, 'フラッシュ'),
    'TenZ': CharacterStats('TenZ', 0.35, 0.22, 85, 0.76, 120.0, 'スモーカー'),
    'おもこ': CharacterStats('おもこ', 0.067, 0.67, 67, 0.67, 67.0, 'フラッシュ'),
    'Primmie': CharacterStats('Primmie', 0.47, 0.24, 59, 0.75, 98.0, 'タイガー'),
    'Derke': CharacterStats('Derke', 0.35, 0.23, 95, 0.7, 110.0, 'フラッシュ'),
    'Meteor': CharacterStats('Meteor', 0.36, 0.28, 73, 0.76, 95.0, 'フラッシュ'),
    'kaajak': CharacterStats('kaajak', 0.35, 0.2, 95, 0.76, 100.0, 'フラッシュ'),
    'Wo0t': CharacterStats('Wo0t', 0.4, 0.16, 78, 0.75, 120.0, 'タイガー'),
    'Smoggy': CharacterStats('Smoggy', 0.38, 0.19, 75, 0.73, 120.0, 'スモーカー'),
    'IbarakiNinja': CharacterStats('IbarakiNinja', 0.28, 0.16, 110, 0.75, 115.0, 'シーカー'),
    'Lar0k': CharacterStats('Lar0k', 0.34, 0.18, 85, 0.73, 123.0, 'タイガー'),
    'Kachina': CharacterStats('Kachina', 0.45, 0.2, 60, 0.8, 90.0, 'タイガー'),
    'Rarga': CharacterStats('Rarga', 0.35, 0.19, 75, 0.75, 120.0, 'タイガー'),
    'HYUNMIN': CharacterStats('HYUNMIN', 0.35, 0.21, 76, 0.75, 110.0, 'フラッシュ'),
    'Zekken': CharacterStats('Zekken', 0.35, 0.22, 85, 0.7, 105.0, 'フラッシュ'),
    'nAts': CharacterStats('nAts', 0.29, 0.18, 125, 0.71, 88.0, 'スモーカー'),
    'Lysoar': CharacterStats('Lysoar', 0.33, 0.17, 105, 0.71, 100.0, 'スモーカー'),
    'ZMJKK': CharacterStats('ZMJKK', 0.3, 0.17, 65, 0.71, 150.0, 'タイガー'),
    'Mazino': CharacterStats('Mazino', 0.33, 0.18, 100, 0.73, 93.0, 'スモーカー'),
    'trent': CharacterStats('trent', 0.3, 0.18, 120, 0.71, 85.0, 'シーカー'),
    'Sato': CharacterStats('Sato', 0.4, 0.19, 67, 0.73, 105.0, 'フラッシュ'),
    'BABYBAY': CharacterStats('BABYBAY', 0.4, 0.15, 85, 0.73, 100.0, 'タイガー'),
    'crashies': CharacterStats('crashies', 0.28, 0.17, 140, 0.68, 80.0, 'シーカー'),
    'Meiy': CharacterStats('Meiy', 0.35, 0.22, 70, 0.73, 100.0, 'タイガー'),
    'Dep': CharacterStats('Dep', 0.33, 0.22, 75, 0.73, 100.0, 'タイガー'),
    'C0M': CharacterStats('C0M', 0.28, 0.16, 120, 0.69, 95.0, 'シーカー'),
    'Buzz': CharacterStats('Buzz', 0.3, 0.22, 80, 0.72, 100.0, 'フラッシュ'),
    'tex': CharacterStats('tex', 0.34, 0.18, 85, 0.73, 95.0, 'フラッシュ'),
    'SugarZ3ro': CharacterStats('SugarZ3ro', 0.29, 0.18, 120, 0.65, 93.0, 'スモーカー'),
    'eggsterr': CharacterStats('eggsterr', 0.3, 0.22, 88, 0.7, 95.0, 'フラッシュ'),
    'skuba': CharacterStats('skuba', 0.31, 0.17, 95, 0.71, 100.0, 'スモーカー'),
    'CHICHOO': CharacterStats('CHICHOO', 0.32, 0.18, 80, 0.74, 100.0, 'スモーカー'),
    'Mako': CharacterStats('Mako', 0.28, 0.16, 120, 0.7, 85.0, 'スモーカー'),
    'Boaster': CharacterStats('Boaster', 0.2, 0.12, 165, 0.62, 96.0, 'スモーカー'),
    'F0rsakeN': CharacterStats('F0rsakeN', 0.26, 0.14, 125, 0.69, 95.0, 'スモーカー'),
    'Boostio': CharacterStats('Boostio', 0.28, 0.15, 128, 0.68, 85.0, 'スモーカー'),
    'stax': CharacterStats('stax', 0.29, 0.15, 130, 0.67, 80.0, 'フラッシュ'),
    'Sayonara': CharacterStats('Sayonara', 0.33, 0.19, 89, 0.7, 90.0, 'シーカー'),
    'keiko': CharacterStats('keiko', 0.31, 0.19, 89, 0.72, 90.0, 'スモーカー'),
    'leaf': CharacterStats('leaf', 0.35, 0.17, 89, 0.74, 80.0, 'フラッシュ'),
    'Rossy': CharacterStats('Rossy', 0.3, 0.18, 89, 0.7, 100.0, 'シーカー'),
    'Verno': CharacterStats('Verno', 0.35, 0.16, 86, 0.7, 95.0, 'シーカー'),
    'd4v41': CharacterStats('d4v41', 0.35, 0.17, 70, 0.71, 105.0, 'シーカー'),
    'valyn': CharacterStats('valyn', 0.25, 0.13, 135, 0.64, 95.0, 'スモーカー'),
    'Jamppi': CharacterStats('Jamppi', 0.25, 0.15, 125, 0.68, 85.0, 'フラッシュ'),
    'Brawk': CharacterStats('Brawk', 0.32, 0.15, 95, 0.69, 95.0, 'シーカー'),
    'Jinggg': CharacterStats('Jinggg', 0.3, 0.23, 68, 0.71, 90.0, 'タイガー'),
    'Rb': CharacterStats('Rb', 0.29, 0.23, 88, 0.67, 80.0, 'スモーカー'),
    'Laz': CharacterStats('Laz', 0.4, 0.14, 70, 0.72, 86.0, 'フラッシュ'),
    'FNS': CharacterStats('FNS', 0.15, 0.13, 170, 0.63, 70.0, 'スモーカー'),
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